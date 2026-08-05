from __future__ import annotations

import json
import os, shutil
import re
import subprocess
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Union, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from app.job_control import JobControl

import ffmpeg
from pydub import AudioSegment


prefix = os.environ.get("CONDA_PREFIX")
ffmpeg_override = os.getenv("FW_FFMPEG")
ffprobe_override = os.getenv("FW_FFPROBE")

_ffmpeg = ffmpeg_override or (
    os.path.join(prefix, "bin", "ffmpeg") if prefix else shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
)
_ffprobe = ffprobe_override or (
    os.path.join(prefix, "bin", "ffprobe") if prefix else shutil.which("ffprobe") or "/usr/bin/ffprobe"
)

AudioSegment.converter = _ffmpeg
AudioSegment.ffprobe = _ffprobe


@dataclass
class _Defaults:
    MAX_AUDIO_BYTES: int = 25 * 1024 * 1024
    DEFAULT_SR: int = 16000
    DEFAULT_BR: str = "96k"
    DEFAULT_CH: int = 1


try:
    from app.config.settings import settings

    _MAX_BYTES = settings.MAX_AUDIO_BYTES
    _SR = settings.DEFAULT_SR
    _BR = settings.DEFAULT_BR
    _CH = settings.DEFAULT_CH
except Exception:
    _MAX_BYTES = _Defaults.MAX_AUDIO_BYTES
    _SR = _Defaults.DEFAULT_SR
    _BR = _Defaults.DEFAULT_BR
    _CH = _Defaults.DEFAULT_CH


class SizeLimitedBuffer(BytesIO):
    """메모리 버퍼에 크기 제한을 두는 Helper 클래스"""

    def __init__(self, limit: int = _MAX_BYTES):
        super().__init__()
        self.limit = int(limit)

    def write(self, b: bytes) -> int:
        if self.tell() + len(b) > self.limit:
            mb = self.limit / (1024**2)
            raise ValueError(f"Buffer size would exceed {mb:.1f} MB")
        return super().write(b)


class AudioProcessor:
    """
    오디오 변환 및 비디오 Demux 기능을 제공하는 클래스
    - 출력은 항상 WAV(PCM16, 16kHz, mono)로 통일
    """

    def __init__(
        self,
        path: Union[str, Path],
        sr: int = _SR,
        br: str = _BR,
        channels: int = _CH,
        max_bytes: int = _MAX_BYTES,
        job_control: "Optional[JobControl]" = None,
    ):
        self.source_audio_path = Path(path) if isinstance(path, str) else path
        self.target_sr = int(sr)
        self.target_br = br
        self.target_channels = int(channels)
        self.max_bytes = int(max_bytes)
        # 취소 시 실행 중인 ffmpeg/ffprobe를 kill하기 위한 훅 (attach_process/detach_process/check)
        self.job_control = job_control

        self.silence_boundaries: Optional[List[Tuple[float, float]]] = None
        self.audio_info: Optional[dict] = None

    def _run_process(self, cmd: List[str], *, capture_stdout: bool = False) -> bytes:
        """서브프로세스 실행. job_control이 있으면 취소 시 즉시 kill 되고 JobCancelled가 발생한다."""
        ctrl = self.job_control
        if ctrl is not None:
            ctrl.check()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if ctrl is not None:
            ctrl.attach_process(proc)
        try:
            stdout, stderr = proc.communicate()
        finally:
            if ctrl is not None:
                ctrl.detach_process()
        if ctrl is not None:
            ctrl.check()  # kill로 종료된 경우 returncode 검사 전에 JobCancelled 발생
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)
        return stdout or b""

    def _export_to_disk(self, data: bytes, stem: str = "audio") -> Path:
        """버퍼 초과/디스크 저장 요청 시 WAV로 파일 저장"""
        output_dir = Path("assets/temp")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{stem}.wav"
        with open(out_path, "wb") as f:
            f.write(data)
        return out_path

    def _write_or_path(self, data: bytes, export_to_disk: bool, stem: str) -> Union[bytes, Path]:
        if export_to_disk or len(data) > self.max_bytes:
            return self._export_to_disk(data, stem=stem)
        return data

    def _clip_segment(self, audio: AudioSegment, start: int, end: int) -> AudioSegment:
        """초 단위 클리핑(디코딩된 AudioSegment에만 적용)"""
        if end > start and start >= 0:
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            if end_ms > start_ms:
                audio = audio[start_ms:end_ms]
        return audio

    def _to_wav_pcm16_bytes(self, audio: AudioSegment) -> bytes:
        audio = audio.set_frame_rate(self.target_sr).set_channels(self.target_channels).set_sample_width(2)
        buf = SizeLimitedBuffer(limit=self.max_bytes)
        audio.export(buf, format="wav")
        return buf.getvalue()

    def get_audio_info(self, get_new_info: bool = False) -> dict:
        if self.audio_info and (not get_new_info):
            return self.audio_info

        path = str(self.source_audio_path)
        cmd = [
            _ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,bit_rate,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            path,
        ]
        info = self._run_process(cmd, capture_stdout=True).decode("utf-8", errors="ignore")
        obj = json.loads(info)

        streams = obj.get("streams", [])
        if not streams:
            raise RuntimeError("No audio stream found.")

        stream_info = streams[0]
        format_info = obj.get("format", {})
        format_dur = float(format_info.get("duration", 0) or 0.0)
        stream_dur = float(stream_info.get("duration", format_dur) or 0.0)
        stream_info["duration"] = stream_dur

        self.audio_info = stream_info
        return stream_info

    def _calculate_split_points(
        self,
        segment_time: int = 3600,
        start: int = 0,
        end: int = 0,
        min_silence_sec: float = 0.5,
        noise: str = "-30dB"
    ) -> List[Tuple[float, float]]:
        a_info = self.get_audio_info()
        total_dur = float(a_info.get("duration", 0.0) or 0.0)
        
        actual_start = float(start)
        actual_end = float(end) if end > start and start >= 0 else total_dur
        
        if actual_end - actual_start <= segment_time:
            return [(actual_start, actual_end)]
            
        silences = self._detect_silence(noise=noise, d=min_silence_sec)
        
        splits = []
        current_start = actual_start
        
        while current_start < actual_end:
            target_end = current_start + segment_time
            if target_end >= actual_end:
                splits.append((current_start, actual_end))
                break
                
            best_split = target_end
            min_diff = float('inf')
            
            for s, e in silences:
                mid = (s + e) / 2.0
                if mid <= current_start:
                    continue
                if mid >= actual_end:
                    continue
                    
                diff = abs(mid - target_end)
                # Only consider silences within a 10-minute window (600 seconds)
                if diff < 600 and diff < min_diff:
                    min_diff = diff
                    best_split = mid
                    
            splits.append((current_start, best_split))
            current_start = best_split
            
        return splits

    def segment(self, segment_time: int = 3600, start: int = 0, end: int = 0) -> List[Path]:
        """
        비디오/오디오에서 오디오를 추출하고 무음 구간을 기준으로 지정된 시간(초) 단위로 분할하여 디스크에 저장합니다.
        메모리를 거의 사용하지 않고(OOM 방지) 빠르게 처리합니다.
        """
        from app.services.temp_files import ensure_temp_dir
        output_dir = ensure_temp_dir()
        stem = self.source_audio_path.stem
        
        split_points = self._calculate_split_points(
            segment_time=segment_time,
            start=start,
            end=end,
            min_silence_sec=0.5
        )
        
        segments = []
        
        for i, (s, e) in enumerate(split_points):
            output_path = output_dir / f"{stem}_{i:03d}.wav"
            cmd = [
                _ffmpeg,
                "-y",
                "-ss", str(s),
                "-t", str(e - s),
                "-i", str(self.source_audio_path),
                "-vn",  # 비디오 스트림 무시
                "-map", "0:a:0",  # 첫 번째 오디오 스트림 선택
                "-acodec", "pcm_s16le",
                "-ac", str(self.target_channels),
                "-ar", str(self.target_sr),
                str(output_path)
            ]
            print(f"[segment] Running ffmpeg for chunk {i}: {' '.join(cmd)}")
            try:
                self._run_process(cmd)
            except subprocess.CalledProcessError as err:
                err_msg = err.stderr.decode("utf-8", errors="ignore") if err.stderr else "Unknown error"
                raise RuntimeError(f"FFmpeg segmentation failed at chunk {i}: {err_msg}")
                
            segments.append(output_path)
            
        return segments

    def extract_wav(self, start: int = 0, end: int = 0) -> Path:
        """
        비디오/오디오에서 전체 구간(또는 start~end 구간)을 16kHz, mono, PCM16 WAV로 추출 (to local)
        """
        from app.services.temp_files import ensure_temp_dir
        output_dir = ensure_temp_dir()
        stem = self.source_audio_path.stem
        output_path = output_dir / f"{stem}_full.wav"
        
        a_info = self.get_audio_info()
        total_dur = float(a_info.get("duration", 0.0) or 0.0)
        actual_start = float(start)
        actual_end = float(end) if end > start and start >= 0 else total_dur
        
        cmd = [
            _ffmpeg,
            "-y",
            "-ss", str(actual_start),
            "-t", str(actual_end - actual_start),
            "-i", str(self.source_audio_path),
            "-vn",  # 비디오 스트림 무시
            "-map", "0:a:0",  # 첫 번째 오디오 스트림 선택
            "-acodec", "pcm_s16le",
            "-ac", str(self.target_channels),
            "-ar", str(self.target_sr),
            str(output_path)
        ]
        print(f"[extract_wav] Running ffmpeg: {' '.join(cmd)}")
        try:
            self._run_process(cmd)
        except subprocess.CalledProcessError as err:
            err_msg = err.stderr.decode("utf-8", errors="ignore") if err.stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg extraction failed: {err_msg}")
            
        return output_path

    def convert(
        self,
        start: int = 0,
        end: int = 0,
        export_to_disk: bool = False,
    ) -> Union[bytes, Path]:
        """
        오디오 파일을 디코딩 후 (필요시) 클리핑 → WAV(PCM16) 바이트/파일로 반환
        - 입력 확장자와 무관하게 처리
        """
        path = self.source_audio_path
        print(f"[convert] input: {path.name}")

        audio = AudioSegment.from_file(path)
        audio = self._clip_segment(audio, start, end)

        wav_bytes = self._to_wav_pcm16_bytes(audio)
        return self._write_or_path(wav_bytes, export_to_disk, stem=path.stem)

    def demux(
        self,
        start: int = 0,
        end: int = 0,
        export_to_disk: bool = False,
    ) -> Union[bytes, Path]:
        """
        비디오에서 오디오만 추출 → WAV(PCM16) 반환
        - ffmpeg pipe로 WAV(pcm_s16le)를 받아 필요 시 클리핑
        """
        path = self.source_audio_path
        print(f"[demux] input: {path.name}")

        proc = (
            ffmpeg.input(str(path))
            .output(
                "pipe:1",
                format="wav",
                acodec="pcm_s16le",
                ac=self.target_channels,
                ar=str(self.target_sr),
            )
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        buf = SizeLimitedBuffer(limit=self.max_bytes)
        try:
            while True:
                chunk = proc.stdout.read(1024 * 1024)
                if not chunk:
                    break
                buf.write(chunk)
        except ValueError:
            proc.kill()
            return self._export_to_disk(buf.getvalue(), stem=path.stem)

        wav_bytes = buf.getvalue()

        if end > start and start >= 0:
            seg = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")
            seg = self._clip_segment(seg, start, end)
            wav_bytes = self._to_wav_pcm16_bytes(seg)

        return self._write_or_path(wav_bytes, export_to_disk, stem=path.stem)

    def _detect_silence(
        self,
        noise: str = "-30dB",
        d: float = 3.0,
        pad: float = 0.3,
        get_new_boundaries: bool = False,
    ) -> List[Tuple[float, float]]:
        """
        FFmpeg `silencedetect` 로그를 파싱해 (silence_start, silence_end) 리스트 반환
        noise : 무음 기준 레벨 (예: '-30dB')
        d     : 무음 판정 최소 지속시간(초)
        pad   : 패딩(초)
        """
        if self.silence_boundaries and (not get_new_boundaries):
            return self.silence_boundaries

        a_info: dict = self.get_audio_info()
        dur: float = float(a_info.get("duration", 0.0) or 0.0)

        _, stderr = (
            ffmpeg.input(str(self.source_audio_path))
            .filter("silencedetect", noise=noise, d=d)
            .output("null", format="null")
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )

        if isinstance(stderr, (bytes, bytearray)):
            stderr = stderr.decode("utf-8", errors="ignore")

        starts = [float(x) for x in re.findall(r"silence_start:\s*([\d.]+)", stderr)]
        ends = [float(x) for x in re.findall(r"silence_end:\s*([\d.]+)", stderr)]

        if len(ends) < len(starts):
            ends.append(dur)

        padded: List[Tuple[float, float]] = []
        for s, e in zip(starts, ends):
            s_pad = max(0.0, s - min(d, pad))
            e_pad = min(dur, e + min(d, pad))
            padded.append((s_pad, e_pad))

        self.silence_boundaries = padded
        return padded

    def find_extended_silence_boundary(
        self,
        ts: float,
        *,
        direction: str = "forward",  # 'forward' or 'backward'
        min_silence_sec: float = 3.0,
        noise: str = "-30dB",
    ) -> float:
        """
        기준 시각 ts(초)에서 direction 방향으로 min_silence_sec 이상 연속된 무음이 있으면
        forward  → 그 무음 **끝 시각**
        backward → 그 무음 **시작 시각** 반환
        없으면 0.0 반환
        """
        if direction not in ("forward", "backward"):
            raise ValueError("direction 은 'forward' 또는 'backward'여야 합니다.")

        intervals = self.silence_boundaries or self._detect_silence(noise=noise, d=min_silence_sec)

        if direction == "forward":
            for s, e in intervals:
                if s >= ts or (s <= ts < e):  # ts 이후 무음 or ts가 무음 내부
                    if e - max(s, ts) >= min_silence_sec:
                        return e
        else:
            for s, e in reversed(intervals):
                if e <= ts or (s < ts <= e):  # ts 이전 무음 or ts가 무음 내부
                    if min(e, ts) - s >= min_silence_sec:
                        return s
        return 0.0
