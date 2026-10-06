using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Spectre.Console;

namespace FractalChat
{
    class Program
    {
        static async Task Main(string[] args)
        {
            await InteractiveModes.StartInteractiveLoopAsync();
        }
    }
}
