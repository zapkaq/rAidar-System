
using Microsoft.AspNetCore.Mvc;
using rAIdar.Services;

namespace rAIdar.Controllers
{
    [Route("api/blockchain")]
    [ApiController]
    public class BlockchainController : ControllerBase
    {
        private readonly MediaHashService _mediaHashService;
        private readonly BlockchainService _blockchainService;

        public BlockchainController(
            MediaHashService mediaHashService,
            BlockchainService blockchainService)
        {
            _mediaHashService = mediaHashService;
            _blockchainService = blockchainService;
        }

        // Generate SHA-256 fingerprint
        [HttpPost("generate-hash")]
        [RequestSizeLimit(100 * 1024 * 1024)]
        public IActionResult GenerateHash(IFormFile file)
        {
            if (file == null || file.Length == 0)
            {
                return BadRequest(new
                {
                    success = false,
                    message = "Please upload a valid media file."
                });
            }

            using (var stream = file.OpenReadStream())
            {
                string hash = _mediaHashService.ComputeSha256(stream);

                return Ok(new
                {
                    success = true,
                    fileName = file.FileName,
                    fileSize = file.Length,
                    algorithm = "SHA-256",
                    hash = hash,
                    message = "Media fingerprint generated successfully."
                });
            }
        }

        // Register media fingerprint on Ethereum Sepolia
        [HttpPost("register")]
        [RequestSizeLimit(100 * 1024 * 1024)]
        public async Task<IActionResult> RegisterMedia(IFormFile file)
        {
            if (file == null || file.Length == 0)
            {
                return BadRequest(new
                {
                    success = false,
                    message = "Please upload a valid media file."
                });
            }

            try
            {
                string hash;

                using (var stream = file.OpenReadStream())
                {
                    hash = _mediaHashService.ComputeSha256(stream);
                }

                string transactionHash =
                    await _blockchainService.RegisterMediaAsync(hash);

                return Ok(new
                {
                    success = true,
                    fileName = file.FileName,
                    algorithm = "SHA-256",
                    hash = hash,
                    transactionHash = transactionHash,
                    transactionUrl =
                        "https://sepolia.etherscan.io/tx/" + transactionHash,
                    message = "Media fingerprint submitted to Ethereum Sepolia."
                });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new
                {
                    success = false,
                    message = "Blockchain registration failed.",
                    error = ex.Message
                });
            }
        }

        // Verify media fingerprint on Ethereum Sepolia
        [HttpPost("verify")]
        [RequestSizeLimit(100 * 1024 * 1024)]
        public async Task<IActionResult> VerifyMedia(IFormFile file)
        {
            if (file == null || file.Length == 0)
            {
                return BadRequest(new
                {
                    success = false,
                    message = "Please upload a valid media file."
                });
            }

            try
            {
                string hash;

                using (var stream = file.OpenReadStream())
                {
                    hash = _mediaHashService.ComputeSha256(stream);
                }

                var result =
                    await _blockchainService.VerifyMediaAsync(hash);

                return Ok(new
                {
                    success = true,
                    fileName = file.FileName,
                    algorithm = "SHA-256",
                    hash = hash,
                    registered = result.Exists,
                    registeredBy = result.Exists
                        ? result.RegisteredBy
                        : null,
                    registeredAt = result.Exists
                        ? DateTimeOffset
                            .FromUnixTimeSeconds((long)result.RegisteredAt)
                            .UtcDateTime
                        : (DateTime?)null,
                    message = result.Exists
                        ? "This media fingerprint is registered on Ethereum."
                        : "This media fingerprint was not found on Ethereum."
                });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new
                {
                    success = false,
                    message = "Blockchain verification failed.",
                    error = ex.Message
                });
            }
        }
    }
}
