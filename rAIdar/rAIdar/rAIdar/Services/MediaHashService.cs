
using System;
using System.IO;
using System.Security.Cryptography;

namespace rAIdar.Services
{
    public class MediaHashService
    {
        public string ComputeSha256(Stream mediaStream)
        {
            if (mediaStream == null)
            {
                throw new ArgumentNullException(nameof(mediaStream));
            }

            if (!mediaStream.CanRead)
            {
                throw new ArgumentException(
                    "The media stream must be readable.",
                    nameof(mediaStream)
                );
            }

            using (SHA256 sha256 = SHA256.Create())
            {
                byte[] hashBytes = sha256.ComputeHash(mediaStream);

                return BitConverter
                    .ToString(hashBytes)
                    .Replace("-", "")
                    .ToLowerInvariant();
            }
        }
    }
}
