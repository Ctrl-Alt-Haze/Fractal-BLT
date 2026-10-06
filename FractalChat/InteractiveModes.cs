using System;
using System.Threading;
using System.Threading.Tasks;
using Spectre.Console;

namespace FractalChat;

public enum UIMode
{
    StandardRAG,
    NativeJITForge,
    HiveMindSwarm
}

public static class InteractiveModes
{
    public static UIMode CurrentMode { get; private set; } = UIMode.StandardRAG;
    private static CancellationTokenSource _cts = new CancellationTokenSource();

    public static async Task StartInteractiveLoopAsync()
    {
        while (true)
        {
            AnsiConsole.Clear();
            RenderHeader();

            _cts = new CancellationTokenSource();
            Task renderTask = CurrentMode switch
            {
                UIMode.HiveMindSwarm => SwarmDashboard.RenderLiveMatrixAsync(_cts.Token),
                UIMode.NativeJITForge => RenderJITForgeAsync(_cts.Token),
                _ => RenderStandardChatAsync(_cts.Token)
            };

            // Wait for hotkey to switch modes
            await Task.Run(() =>
            {
                while (true)
                {
                    if (Console.KeyAvailable)
                    {
                        var key = Console.ReadKey(intercept: true).Key;
                        if (key == ConsoleKey.F1)
                        {
                            SwitchMode(UIMode.StandardRAG);
                            break;
                        }
                        else if (key == ConsoleKey.F2)
                        {
                            SwitchMode(UIMode.NativeJITForge);
                            break;
                        }
                        else if (key == ConsoleKey.F3)
                        {
                            SwitchMode(UIMode.HiveMindSwarm);
                            break;
                        }
                        else if (key == ConsoleKey.Escape)
                        {
                            Environment.Exit(0);
                        }
                    }
                    Thread.Sleep(50);
                }
            });

            await renderTask; // wait for cancellation to finish
        }
    }

    private static void SwitchMode(UIMode newMode)
    {
        if (CurrentMode == newMode) return;
        CurrentMode = newMode;
        _cts.Cancel();
    }

    private static void RenderHeader()
    {
        AnsiConsole.MarkupLine("[grey]Fractal-BLT Interactive TUI // Press F1: Standard Chat | F2: JIT Forge | F3: Hive-Mind | ESC: Exit[/]");
        AnsiConsole.Write(new Rule().RuleStyle("grey"));
    }

    private static async Task RenderStandardChatAsync(CancellationToken token)
    {
        AnsiConsole.MarkupLine("[bold aqua]Standard RAG Chat[/]");
        AnsiConsole.MarkupLine("[grey]Listening to localhost:5000...[/]");
        
        while (!token.IsCancellationRequested)
        {
            await Task.Delay(100, token);
        }
    }

    private static async Task RenderJITForgeAsync(CancellationToken token)
    {
        AnsiConsole.MarkupLine("[bold yellow]Native JIT Forge Monitor[/]");
        AnsiConsole.MarkupLine("[grey]Monitoring tool compilation on unmanaged streams...[/]");

        int tick = 0;
        while (!token.IsCancellationRequested)
        {
            if (tick % 10 == 0)
            {
                AnsiConsole.MarkupLine($"[[{DateTime.Now:HH:mm:ss}]] [green]Forge Standby...[/]");
            }
            await Task.Delay(100, token);
            tick++;
        }
    }
}
