module.exports = {
    apps: [
      {
        name: "Faster-Whsper-API",
        script: "./venv/Scripts/uvicorn.exe", 
        args: "app.main:app --host 0.0.0.0 --port 8000",
        interpreter: "none",
        autorestart: true,
        watch: false,
        max_memory_restart: "10G", // not VRAMs
        // env: {
        // }
      }
    ]
  };