using System;
using System.IO;
using System.Text.Json;
using System.Collections.Generic;

namespace FractalStreamer;

/// <summary>
/// A zero-allocation JSON parser to extract tensor byte offsets from a .safetensors file header.
/// </summary>
public static class SafetensorsHeaderParser
{
    public static bool TryResolveTensor(string path, string tensorName, out string resolvedFilePath, out long offset, out long length)
    {
        resolvedFilePath = path;
        offset = 0;
        length = 0;

        if (Directory.Exists(path))
        {
            string indexPath = Path.Combine(path, "model.safetensors.index.json");
            if (File.Exists(indexPath))
            {
                try
                {
                    string json = File.ReadAllText(indexPath);
                    using JsonDocument doc = JsonDocument.Parse(json);
                    if (doc.RootElement.TryGetProperty("weight_map", out JsonElement weightMap))
                    {
                        if (weightMap.TryGetProperty(tensorName, out JsonElement fileElement))
                        {
                            string fileName = fileElement.GetString() ?? "";
                            resolvedFilePath = Path.Combine(path, fileName);
                            return TryGetTensorOffsets(resolvedFilePath, tensorName, out offset, out length);
                        }
                    }
                }
                catch
                {
                    // Fallback to scanning if index fails
                }
            }

            // Fallback: scan all files
            foreach (var f in Directory.GetFiles(path, "*.safetensors"))
            {
                if (TryGetTensorOffsets(f, tensorName, out offset, out length))
                {
                    resolvedFilePath = f;
                    return true;
                }
            }
            return false;
        }

        if (File.Exists(path))
        {
            return TryGetTensorOffsets(path, tensorName, out offset, out length);
        }

        return false;
    }

    /// <summary>
    /// Parses the safetensors header to find the byte offset and length for a specific tensor name.
    /// This avoids allocating objects or strings, using Utf8JsonReader on a stack or unmanaged buffer.
    /// </summary>
    public static bool TryGetTensorOffsets(string filePath, string tensorName, out long offset, out long length)
    {
        offset = 0;
        length = 0;

        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        
        Span<byte> lengthBuffer = stackalloc byte[8];
        if (fs.Read(lengthBuffer) != 8)
            throw new SafeTensorsParseException("File is too small to contain a SafeTensors length prefix.");

        long headerLength = BitConverter.ToInt64(lengthBuffer);
        if (headerLength <= 0 || headerLength > fs.Length - 8) 
            throw new SafeTensorsParseException($"Invalid SafeTensors header length: {headerLength}. File may be corrupted or truncated.");

        byte[] headerBytes = new byte[headerLength];
        if (fs.Read(headerBytes) != headerLength)
            throw new SafeTensorsParseException("Failed to read the entire SafeTensors JSON header.");

        try
        {
            var reader = new Utf8JsonReader(headerBytes);

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.PropertyName)
                {
                    if (reader.ValueTextEquals(tensorName))
                    {
                        reader.Read(); 
                        if (reader.TokenType != JsonTokenType.StartObject)
                            continue;

                        while (reader.Read() && reader.TokenType != JsonTokenType.EndObject)
                        {
                            if (reader.TokenType == JsonTokenType.PropertyName && reader.ValueTextEquals("data_offsets"))
                            {
                                reader.Read(); 
                                reader.Read(); 
                                long startOffset = reader.GetInt64();
                                reader.Read(); 
                                long endOffset = reader.GetInt64();

                                offset = 8 + headerLength + startOffset;
                                length = endOffset - startOffset;
                                return true;
                            }
                        }
                        throw new SafeTensorsParseException($"Expected 'data_offsets' array for tensor '{tensorName}'.");
                    }
                }
            }
        }
        catch (Exception ex) when (ex is JsonException || ex is InvalidOperationException)
        {
            throw new SafeTensorsParseException("Failed to parse SafeTensors JSON header.", ex);
        }

        return false; // Returns false instead of throwing if tensor is not in this specific file
    }

    /// <summary>
    /// Parses the safetensors header to retrieve all tensor names.
    /// </summary>
    public static List<string> GetAllTensorNames(string path)
    {
        var tensorNames = new List<string>();

        if (Directory.Exists(path))
        {
            foreach (var f in Directory.GetFiles(path, "*.safetensors"))
            {
                tensorNames.AddRange(GetAllTensorNames(f));
            }
            return tensorNames;
        }

        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        
        Span<byte> lengthBuffer = stackalloc byte[8];
        if (fs.Read(lengthBuffer) != 8)
            throw new SafeTensorsParseException("File is too small to contain a SafeTensors length prefix.");

        long headerLength = BitConverter.ToInt64(lengthBuffer);
        if (headerLength <= 0 || headerLength > fs.Length - 8) 
            throw new SafeTensorsParseException($"Invalid SafeTensors header length: {headerLength}. File may be corrupted or truncated.");

        byte[] headerBytes = new byte[headerLength];
        if (fs.Read(headerBytes) != headerLength)
            throw new SafeTensorsParseException("Failed to read the entire SafeTensors JSON header.");

        try
        {
            var reader = new Utf8JsonReader(headerBytes);
            if (!reader.Read() || reader.TokenType != JsonTokenType.StartObject)
                throw new SafeTensorsParseException("Expected a JSON object at the root of the header.");

            while (reader.Read() && reader.TokenType != JsonTokenType.EndObject)
            {
                if (reader.TokenType == JsonTokenType.PropertyName)
                {
                    string propName = reader.GetString() ?? string.Empty;
                    reader.Read(); 
                    
                    if (propName != "__metadata__" && reader.TokenType == JsonTokenType.StartObject)
                    {
                        tensorNames.Add(propName);
                    }
                    
                    reader.Skip(); 
                }
            }
        }
        catch (Exception ex) when (ex is JsonException || ex is InvalidOperationException)
        {
            throw new SafeTensorsParseException("Failed to parse SafeTensors JSON header.", ex);
        }

        return tensorNames;
    }
}
