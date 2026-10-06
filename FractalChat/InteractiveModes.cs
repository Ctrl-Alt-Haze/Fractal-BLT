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

            // We don't await the render task, we let it run in background and manage input here
            var _ = renderTask;
            
            string currentInput = "";
            while (!_cts.Token.IsCancellationRequested)
            {
                if (Console.KeyAvailable)
                {
                    var keyInfo = Console.ReadKey(intercept: true);
                    var key = keyInfo.Key;
                    
                    if (key == ConsoleKey.F1)
                    {
                        SwitchMode(UIMode.StandardRAG);
                    }
                    else if (key == ConsoleKey.F2)
                    {
                        SwitchMode(UIMode.NativeJITForge);
                    }
                    else if (key == ConsoleKey.F3)
                    {
                        SwitchMode(UIMode.HiveMindSwarm);
                    }
                    else if (key == ConsoleKey.Escape)
                    {
                        Environment.Exit(0);
                    }
                    else if (CurrentMode == UIMode.StandardRAG)
                    {
                        if (key == ConsoleKey.Enter)
                        {
                            Console.WriteLine();
                            if (!string.IsNullOrWhiteSpace(currentInput))
                            {
                                string prompt = currentInput;
                                AnsiConsole.MarkupLine($"[grey]Ingested >> {prompt}[/]");
                                currentInput = "";
                                
                                // Fire and forget the response simulation
                                _ = Task.Run(async () => {
                                    await StreamRealHiveMindResponseAsync(prompt);
                                    AnsiConsole.Markup("\n[green]You:[/] ");
                                });
                            }
                            else 
                            {
                                AnsiConsole.Markup("\n[green]You:[/] ");
                            }
                        }
                        else if (key == ConsoleKey.Backspace && currentInput.Length > 0)
                        {
                            currentInput = currentInput.Substring(0, currentInput.Length - 1);
                            Console.Write("\b \b");
                        }
                        else if (!char.IsControl(keyInfo.KeyChar))
                        {
                            currentInput += keyInfo.KeyChar;
                            Console.Write(keyInfo.KeyChar);
                        }
                    }
                }
                await Task.Delay(20);
            }

            try { await renderTask; } catch (TaskCanceledException) { }
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
        AnsiConsole.MarkupLine("[grey]Type your prompt below. Connecting to SwarmBus...[/]");
        AnsiConsole.Markup("[green]You:[/] ");
        
        try 
        {
            while (!token.IsCancellationRequested)
            {
                await Task.Delay(100, token);
            }
        }
        catch (TaskCanceledException) {}
    }

    private static async Task RenderJITForgeAsync(CancellationToken token)
    {
        AnsiConsole.MarkupLine("[bold yellow]Native JIT Forge Monitor[/]");
        AnsiConsole.MarkupLine("[grey]Monitoring tool compilation on unmanaged streams...[/]");

        int tick = 0;
        try
        {
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
        catch (TaskCanceledException) {}
    }

    private static async Task StreamRealHiveMindResponseAsync(string prompt)
    {
        AnsiConsole.Markup("[bold magenta]Hive-Mind:[/] ");
        try
        {
            using var client = new HttpClient();
            client.Timeout = TimeSpan.FromMinutes(5);
            var requestBody = new 
            { 
                model = "qwen3_30b_a3b", 
                messages = new[] { new { role = "user", content = prompt } },
                stream = true 
            };
            
            var content = new StringContent(System.Text.Json.JsonSerializer.Serialize(requestBody), System.Text.Encoding.UTF8, "application/json");
            
            var request = new HttpRequestMessage(HttpMethod.Post, "http://localhost:5000/v1/chat/completions");
            request.Content = content;

            using var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
            response.EnsureSuccessStatusCode();

            using var stream = await response.Content.ReadAsStreamAsync();
            using var reader = new System.IO.StreamReader(stream);

            while (!reader.EndOfStream)
            {
                var line = await reader.ReadLineAsync();
                if (string.IsNullOrWhiteSpace(line)) continue;
                
                if (line.StartsWith("data: "))
                {
                    string data = line.Substring(6);
                    if (data == "[DONE]") break;
                    
                    try 
                    {
                        using var doc = System.Text.Json.JsonDocument.Parse(data);
                        var delta = doc.RootElement.GetProperty("choices")[0].GetProperty("delta");
                        if (delta.TryGetProperty("content", out var contentProp))
                        {
                            Console.Write(contentProp.GetString());
                        }
                    }
                    catch { /* ignore parse errors for raw stream drops */ }
                }
            }
            Console.WriteLine();
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Connection Error: Is FractalServe running on localhost:5000? ({ex.Message})[/]");
        }
    }
}
