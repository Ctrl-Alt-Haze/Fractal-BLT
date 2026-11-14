using System;
using System.IO;
using System.IO.Pipelines;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using CoreLoop;

namespace FractalServe;

public class Program
{
    private static readonly SemaphoreSlim _generationLock = new SemaphoreSlim(1, 1);
    
    private static Tokenizer s_tokenizer = null!;
    private static KVCache s_kvCache = null!;
    private static IGenerator s_generator = null!;
    private static CoreLoop.MetaSteeringAgent s_metaAgent = null!;
    private static CoreLoop.SwarmCoordinator s_swarm = null!;
    private static nint s_cudaContext;
    private static nint s_cublasHandle;
    private static bool s_isMoe = false;

    public static async Task Main(string[] args)
    {
        Console.WriteLine("=== FRACTAL-SERVE INIT ===");
        Console.WriteLine("Architecture: NativeAOT SSE Streaming Pipeline + MoE Blackwell + Autonomous Swarm\n");

        // 1. Initialize Bare-Metal GPU Hardware & Thread Context
        CudaInterop.cuInit(0);
        CudaInterop.cuCtxCreate(out s_cudaContext, 0, 0);
        
        // CRITICAL: Bind context to the thread before creating the cuBLAS handle
        CudaInterop.cuCtxSetCurrent(s_cudaContext);
        
        int cublasStatus = CublasInterop.cublasCreate(out s_cublasHandle);
        if (cublasStatus != 0)
        {
            Console.WriteLine($"[cuBLAS FATAL] cublasCreate failed with status code: {cublasStatus}");
        }
        else
        {
            Console.WriteLine("[cuBLAS] Handle created and bound successfully.");
        }

        // 2. Hydrate Engine Singletons
        string moeDir = @"C:\Fractal-BLT\Models\qwen3_30b_modular";
        if (Directory.Exists(moeDir) && File.Exists(Path.Combine(moeDir, "submodel_00_embeddings.safetensors")))
        {
            Console.WriteLine("\n[FractalServe] >>> ACTIVATING SUPER-AI: QWEN3-30B-A3B MODULAR MoE <<<");
            s_isMoe = true;
            s_tokenizer = new Tokenizer(Path.Combine(moeDir, "tokenizer.json"));

    s_kvCache = new KVCache(numLayers: 48, kvDim: 512, maxSeqLen: 2048);
    s_kvCache.Allocate(s_cudaContext);

    var moeBrain = new MoeBrain();
    await moeBrain.InitializeAndLoadAsync(moeDir, s_cudaContext);

    var moeGen = new MoeGenerator(moeBrain, s_kvCache, s_cudaContext, s_cublasHandle);
    await moeGen.InitializeKernelsAsync();

    s_generator = moeGen;
    s_metaAgent = new CoreLoop.MetaSteeringAgent(s_generator);
    s_swarm = new CoreLoop.SwarmCoordinator(s_generator, s_tokenizer, s_metaAgent);

    s_swarm.PrewarmSnapshots();
}
        else
        {
            Console.WriteLine("\n[FractalServe] Defaulting to Dense Model Engine...");
            s_tokenizer = new Tokenizer(Path.Combine(Environment.CurrentDirectory, "Models", "tokenizer.json"));
            
            s_kvCache = new KVCache();
            s_kvCache.Allocate(s_cudaContext);

            var brain = new Brain();
            await brain.LoadToVRAMAsync(s_cudaContext);

            var gen = new Generator(brain, s_kvCache, s_cudaContext, s_cublasHandle);
            await gen.InitializeKernelsAsync();
            
            s_generator = gen;
            s_metaAgent = new CoreLoop.MetaSteeringAgent(s_generator);
            s_swarm = new CoreLoop.SwarmCoordinator(s_generator, s_tokenizer, s_metaAgent);
        }

        Console.WriteLine("\n[FractalServe] Swarm Hive-Mind fused. Kestrel Swarm ready for incoming API connections.");

        var builder = WebApplication.CreateSlimBuilder(args);
        
        builder.Services.ConfigureHttpJsonOptions(options =>
        {
            options.SerializerOptions.TypeInfoResolverChain.Insert(0, FractalJsonContext.Default);
        });

        builder.Services.AddCors(options =>
        {
            options.AddPolicy("AllowAll", policy => policy.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader());
        });

        var app = builder.Build();

        app.Lifetime.ApplicationStopping.Register(() => 
        {
            Console.WriteLine("\n[Shutdown] Flushing VRAM and destroying CUDA Contexts...");
            s_kvCache?.Free();
            if (s_cudaContext != IntPtr.Zero) CudaInterop.cuCtxDestroy(s_cudaContext);
        });

        app.UseCors("AllowAll");
        app.UseDefaultFiles();
        app.UseStaticFiles();

        app.MapPost("/v1/chat/completions", async (HttpContext context, ChatCompletionRequest request) =>
        {
            await _generationLock.WaitAsync();
            try
            {
                int seqPos = 0;
                int currentToken = 0;

                string lastUserMsg = "";
                if (request.Messages != null)
                {
                    for (int i = request.Messages.Count - 1; i >= 0; i--)
                    {
                        if (request.Messages[i].Role?.ToLower() == "user")
                        {
                            lastUserMsg = request.Messages[i].Content ?? "";
                            break;
                        }
                    }
                }

                var intent = s_swarm.RouteIntent(lastUserMsg);
                Console.WriteLine($"\n[SwarmRouter] User Query: \"{lastUserMsg}\" -> Routed Intent: {intent}");

                if (s_isMoe)
                {
                    s_swarm.PrepareContext(intent, out seqPos);

                    var sb = new StringBuilder();
                    if (request.Messages != null)
                    {
                        foreach (var m in request.Messages)
                        {
                            if (m.Role != null && m.Role.ToLower() == "system") continue;
                            sb.Append($"<|im_start|>{m.Role}\n{m.Content}<|im_end|>\n");
                        }
                    }
                    sb.Append("<|im_start|>assistant\n");

                    string prompt = sb.ToString();
                    int[] userTokens = s_tokenizer.Encode(prompt);

                    currentToken = userTokens[0];
                    for (int i = 0; i < userTokens.Length - 1; i++)
                    {
                        s_generator.PrefillToken(userTokens[i], seqPos);
                        seqPos++;
                        currentToken = userTokens[i + 1];
                    }
                }
                else
                {
                    s_generator.ResetState();

                    var sb = new StringBuilder();
                    sb.Append("The following is a conversation between a Human and a brilliant AI Assistant.\n\n");
                    if (request.Messages != null)
                    {
                        foreach (var m in request.Messages)
                        {
                            if (m.Role != null && m.Role.ToLower() == "user") sb.Append($"Human: {m.Content}\n"); else sb.Append($"Assistant: {m.Content}\n"); 
                        }
                    }
                    sb.Append("Assistant:");

                    string prompt = sb.ToString();
                    int[] promptTokens = s_tokenizer.Encode(prompt);

                    currentToken = promptTokens[0];
                    for (int i = 0; i < promptTokens.Length - 1; i++)
                    {
                        s_generator.PrefillToken(promptTokens[i], seqPos);
                        seqPos++;
                        currentToken = promptTokens[i + 1];
                    }
                }

                if (request.Stream == true)
                {
                    context.Response.ContentType = "text/event-stream";
                    context.Response.Headers.CacheControl = "no-cache";
                    context.Response.Headers.Connection = "keep-alive";

                    var writer = context.Response.BodyWriter;

                    string generatedText = "";
                    for (int i = 0; i < 2000; i++)
                    {
                        int nextToken = s_metaAgent.SteerNextToken(currentToken, seqPos);
                        
                        string word = s_tokenizer.Decode(new[] { nextToken });
                        generatedText += word;
                        
                        if (nextToken == 151645 || nextToken == 151643 || (!s_isMoe && generatedText.Contains("\nHuman:"))) 
                            break;

                        await StreamChunkAsync(writer, word);
                        
                        currentToken = nextToken;
                        seqPos++;
                        
                        await Task.Yield();
                    }

                    await writer.WriteAsync(Encoding.UTF8.GetBytes("data: [DONE]\n\n"));
                    await writer.FlushAsync();
                    
                    return Results.Empty;
                }
                else
                {
                    return Results.BadRequest("Only stream=true is supported in the zero-allocation harness.");
                }
            }
            finally
            {
                _generationLock.Release();
            }
        });

        await app.RunAsync();
    }

    private static async Task StreamChunkAsync(PipeWriter writer, string tokenStr)
    {
        if (string.IsNullOrEmpty(tokenStr)) return;

        var chunk = new ChatCompletionChunk
        {
            Choices = new System.Collections.Generic.List<ChatCompletionChunkChoice>
            {
                new ChatCompletionChunkChoice
                {
                    Delta = new ChatCompletionDelta
                    {
                        Content = tokenStr
                    }
                }
            }
        };

        await writer.WriteAsync(Encoding.UTF8.GetBytes("data: "));
        JsonSerializer.Serialize(writer.AsStream(), chunk, FractalJsonContext.Default.ChatCompletionChunk);
        await writer.WriteAsync(Encoding.UTF8.GetBytes("\n\n"));
        await writer.FlushAsync();
    }
}
