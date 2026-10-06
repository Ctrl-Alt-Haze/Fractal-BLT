---
layout: default
title: Hermes SDK Reference
---

# 🔌 Hermes SDK Integration Guide

The Hermes SDK provides a developer-friendly C# wrapper to connect external applications directly to the localized NativeAOT Swarm.

## NuGet Installation
Include the `Opure.Hermes.SDK` package in your project.

## Basic Usage

The SDK bypasses traditional REST/HTTP overhead by utilizing unmanaged Named Pipes directly connected to the `SwarmBus` physical memory.

```csharp
using FractalBLT.SDK;
using System.Threading.Tasks;

public async Task SendPromptAsync()
{
    using var client = new HermesClient();
    
    // Connect to the SwarmBus Named Pipe
    await client.ConnectAsync();
    
    // Submit prompt (Zero-allocation memory copy directly to the Router Agent)
    bool success = await client.SubmitPromptAsync("Write a C# script to reverse a binary tree.");
    
    if (success)
    {
        Console.WriteLine("Prompt ingested by the Hive-Mind.");
    }
}
```

## How It Works
- **Zero GC Serialization:** Prompts are serialized directly to unmanaged structs (`HermesPacket`) without managed array overhead.
- **Microsecond Latency:** Bypasses TCP stack entirely.
