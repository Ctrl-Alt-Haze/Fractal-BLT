using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Spectre.Console;
using Spectre.Console.Rendering;

namespace FractalChat;

public class ChatMessage
{
    public string role { get; set; } = "";
    public string content { get; set; } = "";
}

public static class InteractiveModes
{
    private static List<ChatMessage> _messages = new();

    private static string FormatMarkdown(string raw)
    {
        if (string.IsNullOrEmpty(raw)) return "";

        // Scrub any stray raw model artifacts
        raw = raw.Replace("<|im_end|>", "")
                 .Replace("<|im_start|>", "")
                 .Replace("<|repo_name|>", "")
                 .Replace("<|endoftext|>", "");

        var escaped = Markup.Escape(raw);
        bool inCode = false;
        var lines = escaped.Split('\n');
        for (int i = 0; i < lines.Length; i++)
        {
            string trimmed = lines[i].TrimStart();
            if (trimmed.StartsWith("```"))
            {
                if (!inCode)
                {
                    string lang = trimmed.Length > 3 ? trimmed.Substring(3).Trim() : "";
                    string badge = string.IsNullOrWhiteSpace(lang) ? "CODE" : lang.ToUpperInvariant();
                    lines[i] = $"\n[grey]┌─── [/][bold gold1]{badge}[/][grey] ───────────────────────────────────────────[/]\n[bold turquoise2]";
                    inCode = true;
                }
                else
                {
                    lines[i] = "[/]\n[grey]└────────────────────────────────────────────────────────┘[/]\n";
                    inCode = false;
                }
            }
        }

        string result = string.Join("\n", lines) + (inCode ? "[/]" : "");
        return result;
    }

    public static async Task StartInteractiveLoopAsync()
    {
        AnsiConsole.Clear();

        var grid = new Grid();
        grid.AddColumn(new GridColumn().NoWrap().PadRight(4));
        grid.AddColumn(new GridColumn().NoWrap().PadRight(4));
        grid.AddColumn(new GridColumn().NoWrap().PadRight(4));
        grid.AddColumn(new GridColumn().NoWrap());
        
        grid.AddRow("[bold cyan]ENGINE[/]", "[bold cyan]SUPER-AI ARCHITECTURE[/]", "[bold cyan]SWARM HIVE-MIND[/]", "[bold cyan]LATENCY[/]");
        grid.AddRow(
            "[white]OpureRuntime (NativeAOT)[/]", 
            "[yellow]Qwen3-30B MoE (128 Experts)[/]", 
            "[green]Fused (Router, Jev, Critic, Tool-Forger)[/]", 
            "[white]< 1ms DMA[/]"
        );

        var headerPanel = new Panel(grid)
            .Border(BoxBorder.Rounded)
            .Header(new PanelHeader("[bold white] FRACTAL-BLT APOLLO CHAT & SWARM HIVE-MIND v2.5 [/]", Justify.Center))
            .BorderColor(Color.Cyan1)
            .Expand();
            
        AnsiConsole.Write(headerPanel);
        AnsiConsole.WriteLine();

        while (true)
        {
            var prompt = AnsiConsole.Prompt(
                new TextPrompt<string>("[bold springgreen2]>[/] ")
                    .PromptStyle("white")
            );

            if (string.IsNullOrWhiteSpace(prompt)) continue;
            if (prompt.Trim().ToLower() == "/exit" || prompt.Trim().ToLower() == "/quit") break;
            if (prompt.Trim().ToLower() == "/clear")
            {
                AnsiConsole.Clear();
                _messages.Clear();
                AnsiConsole.Write(headerPanel);
                continue;
            }

            _messages.Add(new ChatMessage { role = "user", content = prompt });

            var getAssistantPanel = (string content, string metrics) => 
            {
                var p = new Panel(new Markup(content))
                    .Header(string.IsNullOrEmpty(metrics) 
                        ? "[bold mediumpurple2] FRACTAL SWARM [/]" 
                        : $"[bold mediumpurple2] FRACTAL SWARM [/] [grey]({metrics})[/]")
                    .BorderColor(Color.MediumPurple2)
                    .Padding(2, 1, 2, 1)
                    .RoundedBorder();
                return p;
            };

            await AnsiConsole.Live(getAssistantPanel("[grey]Fusing Tensors & Routing Swarm...[/]", ""))
                .StartAsync(async ctx =>
                {
                    string fullResponse = "";
                    var sw = Stopwatch.StartNew();
                    int tokenCount = 0;
                    bool firstToken = false;
                    long ttft = 0;

                    try
                    {
                        using var client = new HttpClient();
                        client.Timeout = TimeSpan.FromMinutes(10);
                        
                        var requestBody = new 
                        { 
                            model = "opure-runtime", 
                            messages = _messages.ToArray(),
                            stream = true 
                        };
                        
                        var content = new StringContent(JsonSerializer.Serialize(requestBody), Encoding.UTF8, "application/json");
                        var request = new HttpRequestMessage(HttpMethod.Post, "http://localhost:5000/v1/chat/completions") { Content = content };

                        using var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
                        response.EnsureSuccessStatusCode();

                        using var stream = await response.Content.ReadAsStreamAsync();
                        using var reader = new System.IO.StreamReader(stream);

                        while (true)
                        {
                            var line = await reader.ReadLineAsync();
                            if (line == null) break;
                            if (string.IsNullOrWhiteSpace(line)) continue;
                            
                            if (line.StartsWith("data: "))
                            {
                                string data = line.Substring(6);
                                if (data == "[DONE]") break;
                                
                                try 
                                {
                                    using var doc = JsonDocument.Parse(data);
                                    var delta = doc.RootElement.GetProperty("choices")[0].GetProperty("delta");
                                    if (delta.TryGetProperty("content", out var contentProp))
                                    {
                                        if (!firstToken) { ttft = sw.ElapsedMilliseconds; firstToken = true; }
                                        fullResponse += contentProp.GetString();
                                        tokenCount++;
                                        
                                        string metrics = $"TTFT: {ttft}ms";
                                        ctx.UpdateTarget(getAssistantPanel($"[white]{FormatMarkdown(fullResponse)}[/]", metrics));
                                        ctx.Refresh();
                                    }
                                }
                                catch { }
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        fullResponse += $"\n\n[red]Error: {Markup.Escape(ex.Message)}[/]";
                        ctx.UpdateTarget(getAssistantPanel(fullResponse, "ERROR"));
                        ctx.Refresh();
                    }

                    sw.Stop();
                    double tps = tokenCount > 0 ? (tokenCount / sw.Elapsed.TotalSeconds) : 0;
                    string finalMetrics = $"TTFT: {ttft}ms | Speed: {tps:F1} tok/s | Tokens: {tokenCount}";
                    
                    ctx.UpdateTarget(getAssistantPanel($"[white]{FormatMarkdown(fullResponse)}[/]", finalMetrics));
                    
                    _messages.Add(new ChatMessage { role = "assistant", content = fullResponse });
                });
                
            AnsiConsole.WriteLine();
        }
    }
}
