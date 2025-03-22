module.exports = async (kernel) => {
  let cmd = "uv pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu"
  if (kernel.platform === 'darwin') {
    if (kernel.arch === "arm64") {
      cmd = "uv pip install torch torchaudio torchvision"
    } else {
      cmd = "uv pip install torch==2.1.2 torchaudio==2.1.2"
    }
  } else {
    if (kernel.gpu === 'nvidia') {
      if (kernel.gpu_model && / 50.+/.test(kernel.gpu_model)) {
        cmd = "uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128"
      } else {
        cmd = "uv pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 xformers --index-url https://download.pytorch.org/whl/cu124"
      }
    } else if (kernel.gpu === 'amd') {
      cmd = "uv pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/rocm6.2"
    } 
  }
//module.exports = {
//  "cmds": {
//    "nvidia": "pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 xformers --index-url https://download.pytorch.org/whl/cu121",
//    "amd": "pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/rocm6.0",
//    "default": "pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu"
//  },
//  "requires": [{
//    "type": "conda",
//    "name": "ffmpeg",
//    "args": "-c conda-forge"
//  }],
  return {
    "run": [{
      "method": "shell.run",
      "params": {
        "venv": "env",
        "message": [
          "pip install -r requirements.txt",
          "{{(gpu === 'nvidia' ? self.cmds.nvidia : (gpu === 'amd' ? self.cmds.amd : self.cmds.default))}}"
        ]
      }
    }, {
      "method": "notify",
      "params": {
        "html": "Click the 'start' tab to get started!"
      }
    }]
  }
}
