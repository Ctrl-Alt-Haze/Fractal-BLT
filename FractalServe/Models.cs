using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace FractalServe;

public struct ChatCompletionMessage
{
    [JsonPropertyName("role")]
    public string Role { get; set; }

    [JsonPropertyName("content")]
    public string Content { get; set; }
}

public struct ChatCompletionRequest
{
    [JsonPropertyName("model")]
    public string Model { get; set; }

    [JsonPropertyName("messages")]
    public List<ChatCompletionMessage> Messages { get; set; }

    [JsonPropertyName("stream")]
    public bool? Stream { get; set; }
}

public struct ChatCompletionChoice
{
    [JsonPropertyName("message")]
    public ChatCompletionMessage Message { get; set; }
}

public struct ChatCompletionResponse
{
    [JsonPropertyName("choices")]
    public List<ChatCompletionChoice> Choices { get; set; }
}

public struct ChatCompletionDelta
{
    [JsonPropertyName("content")]
    public string Content { get; set; }
}

public struct ChatCompletionChunkChoice
{
    [JsonPropertyName("delta")]
    public ChatCompletionDelta Delta { get; set; }
}

public struct ChatCompletionChunk
{
    [JsonPropertyName("choices")]
    public List<ChatCompletionChunkChoice> Choices { get; set; }
}

public struct ExperimentalStubResponse
{
    [JsonPropertyName("error")]
    public string Error { get; set; }

    [JsonPropertyName("message")]
    public string Message { get; set; }
}

[JsonSerializable(typeof(ExperimentalStubResponse))]
[JsonSerializable(typeof(ChatCompletionRequest))]
[JsonSerializable(typeof(ChatCompletionMessage))]
[JsonSerializable(typeof(ChatCompletionChoice))]
[JsonSerializable(typeof(ChatCompletionResponse))]
[JsonSerializable(typeof(ChatCompletionChunk))]
[JsonSerializable(typeof(ChatCompletionChunkChoice))]
[JsonSerializable(typeof(ChatCompletionDelta))]
[JsonSerializable(typeof(List<ChatCompletionChoice>))]
[JsonSerializable(typeof(List<ChatCompletionMessage>))]
[JsonSerializable(typeof(List<ChatCompletionChunkChoice>))]
public partial class FractalJsonContext : JsonSerializerContext
{
}