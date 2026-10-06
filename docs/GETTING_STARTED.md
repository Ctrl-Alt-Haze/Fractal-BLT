---
layout: default
title: Getting Started
---

# 🚀 Getting Started with Fractal-BLT

Fractal-BLT is a massive NativeAOT C# leap. It configures itself effortlessly.

## 1. 1-Click Deployment (OZTAE Bootstrapper)
Run the Windows PowerShell bootstrapper from the project root:
```powershell
./Installer/Forge-Fractal.ps1
```

**What this does:**
1. Triggers `dotnet publish -c Release -r win-x64 /p:PublishAot=true` to compile the entire ecosystem down to native metal (no CLR required).
2. Locks the executable inside an **Opure Zero-Trust Agent Enclave (OZTAE)** using Windows Job Objects (strict 2GB memory bound).
3. Evaluates your `Models/` directory. If it only sees `qwen3_30b_a3b`, it fires up the **Model Harvester** to automatically pull a 1.5B traffic cop model for Speculative Decoding.

## 2. Booting the Thought-Matrix TUI
Once compiled, start the environment. The **Gemini Thought-Matrix TUI** (Spectre.Console) will launch.
Use your hotkeys:
- `F1`: Standard RAG Chat
- `F2`: JIT Forge Monitor
- `F3`: Hive-Mind Swarm Matrix (4-lane live telemetry)
- `ESC`: Exit
