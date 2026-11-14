namespace FractalServe;

/// <summary>
/// Marks a type whose public behavior is a demo or an unfinished algorithm.
/// The attribute is documentation for readers and does not change runtime behavior.
/// </summary>
[AttributeUsage(AttributeTargets.Class | AttributeTargets.Method | AttributeTargets.Struct)]
public sealed class ExperimentalStubAttribute : Attribute
{
    public ExperimentalStubAttribute(string reason) => Reason = reason;

    public string Reason { get; }
}
