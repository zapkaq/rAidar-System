
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using rAIdar.Data;
using rAIdar.Models;
using rAIdar.Services;
using System.Net.Http.Headers;
using System.Text.Json;

namespace rAIdar.Controllers
{
    public class DetectController : Controller
    {
        private readonly IWebHostEnvironment _environment;
        private readonly ApplicationDbContext _context;
        private readonly MediaHashService _mediaHashService;
        private readonly BlockchainService _blockchainService;

        private static readonly HashSet<string> ImageExtensions =
            new(StringComparer.OrdinalIgnoreCase)
            {
                ".jpg", ".jpeg", ".png"
            };

        private static readonly HashSet<string> VideoExtensions =
            new(StringComparer.OrdinalIgnoreCase)
            {
                ".mp4", ".mov", ".avi", ".mkv", ".webm"
            };

        private static readonly HashSet<string> AudioExtensions =
            new(StringComparer.OrdinalIgnoreCase)
            {
                ".wav", ".mp3", ".flac", ".ogg", ".m4a"
            };

        public DetectController(
            IWebHostEnvironment environment,
            ApplicationDbContext context,
            MediaHashService mediaHashService,
            BlockchainService blockchainService)
        {
            _environment = environment;
            _context = context;
            _mediaHashService = mediaHashService;
            _blockchainService = blockchainService;
        }

        // Register the latest analyzed media fingerprint on Ethereum.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> RegisterFingerprint()
        {
            int? userId = HttpContext.Session.GetInt32("UserId");

            if (userId == null)
            {
                return Unauthorized(new
                {
                    success = false,
                    message = "Please log in again before registering media."
                });
            }

            string? mediaHash =
                HttpContext.Session.GetString("LastAnalyzedMediaHash");

            if (string.IsNullOrWhiteSpace(mediaHash))
            {
                return BadRequest(new
                {
                    success = false,
                    message = "No recent media fingerprint was found. Analyze the media again."
                });
            }

            try
            {
                var histories = await _context.DetectionHistories
                    .Where(h =>
                        h.UserId == userId.Value &&
                        h.MediaHash == mediaHash)
                    .ToListAsync();

                if (histories.Count == 0)
                {
                    return BadRequest(new
                    {
                        success = false,
                        message = "The detection history for this media was not found. Please analyze the media again."
                    });
                }

                string? existingTransactionHash = histories
                    .Select(h => h.BlockchainTransactionHash)
                    .FirstOrDefault(hash =>
                        !string.IsNullOrWhiteSpace(hash));

                var existingRecord =
                    await _blockchainService.VerifyMediaAsync(mediaHash);

                if (existingRecord.Exists)
                {
                    DateTime registeredAtUtc =
                        DateTimeOffset
                            .FromUnixTimeSeconds(
                                (long)existingRecord.RegisteredAt)
                            .UtcDateTime;

                    foreach (var history in histories)
                    {
                        history.BlockchainRegistered = true;
                        history.BlockchainRegisteredBy =
                            existingRecord.RegisteredBy;
                        history.BlockchainRegisteredAtUtc =
                            registeredAtUtc;

                        if (!string.IsNullOrWhiteSpace(existingTransactionHash))
                        {
                            history.BlockchainTransactionHash =
                                existingTransactionHash;
                        }
                    }

                    await _context.SaveChangesAsync();

                    return Conflict(new
                    {
                        success = false,
                        alreadyRegistered = true,
                        registeredBy = existingRecord.RegisteredBy,
                        registeredAtUtc =
                            registeredAtUtc.ToString("yyyy-MM-dd HH:mm:ss"),
                        message = "This media fingerprint is already registered on Ethereum."
                    });
                }

                string transactionHash =
                    await _blockchainService.RegisterMediaAsync(mediaHash);

                var registeredRecord =
                    await _blockchainService.VerifyMediaAsync(mediaHash);

                if (!registeredRecord.Exists)
                {
                    return StatusCode(500, new
                    {
                        success = false,
                        message = "The transaction was submitted, but the registered record could not be confirmed."
                    });
                }

                DateTime newlyRegisteredAtUtc =
                    DateTimeOffset
                        .FromUnixTimeSeconds(
                            (long)registeredRecord.RegisteredAt)
                        .UtcDateTime;

                foreach (var history in histories)
                {
                    history.BlockchainRegistered = true;
                    history.BlockchainTransactionHash = transactionHash;
                    history.BlockchainRegisteredBy =
                        registeredRecord.RegisteredBy;
                    history.BlockchainRegisteredAtUtc =
                        newlyRegisteredAtUtc;
                }

                await _context.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    registered = true,
                    registeredBy = registeredRecord.RegisteredBy,
                    registeredAtUtc =
                        newlyRegisteredAtUtc.ToString("yyyy-MM-dd HH:mm:ss"),
                    transactionHash = transactionHash,
                    transactionUrl =
                        "https://sepolia.etherscan.io/tx/" + transactionHash,
                    message = "Media fingerprint successfully registered on Ethereum Sepolia."
                });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new
                {
                    success = false,
                    message = ex.Message,
                    detail = ex.InnerException?.Message
                });
            }
        }

        // Recover the original Ethereum transaction hash for an existing fingerprint.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> RecoverTransactionHash(string mediaHash)
        {
            int? userId = HttpContext.Session.GetInt32("UserId");

            if (userId == null)
            {
                return Unauthorized(new
                {
                    success = false,
                    message = "Please log in again."
                });
            }

            if (string.IsNullOrWhiteSpace(mediaHash) ||
                mediaHash.Length != 64)
            {
                return BadRequest(new
                {
                    success = false,
                    message = "Invalid SHA-256 fingerprint."
                });
            }

            try
            {
                var histories = await _context.DetectionHistories
                    .Where(h =>
                        h.UserId == userId.Value &&
                        h.MediaHash == mediaHash)
                    .ToListAsync();

                if (histories.Count == 0)
                {
                    return NotFound(new
                    {
                        success = false,
                        message = "No matching detection history was found."
                    });
                }

                var blockchainRecord =
                    await _blockchainService.VerifyMediaAsync(mediaHash);

                if (!blockchainRecord.Exists)
                {
                    return Conflict(new
                    {
                        success = false,
                        message = "This fingerprint is not registered on Ethereum."
                    });
                }

                string? transactionHash =
                    await _blockchainService.FindRegistrationTransactionAsync(
                        mediaHash);

                if (string.IsNullOrWhiteSpace(transactionHash))
                {
                    return NotFound(new
                    {
                        success = false,
                        message = "No matching registration transaction was found."
                    });
                }

                DateTime registeredAtUtc =
                    DateTimeOffset
                        .FromUnixTimeSeconds(
                            (long)blockchainRecord.RegisteredAt)
                        .UtcDateTime;

                foreach (var history in histories)
                {
                    history.BlockchainRegistered = true;
                    history.BlockchainTransactionHash = transactionHash;
                    history.BlockchainRegisteredBy =
                        blockchainRecord.RegisteredBy;
                    history.BlockchainRegisteredAtUtc =
                        registeredAtUtc;
                }

                await _context.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    transactionHash = transactionHash,
                    transactionUrl =
                        "https://sepolia.etherscan.io/tx/" + transactionHash,
                    updatedRecords = histories.Count,
                    message = "Original blockchain transaction successfully recovered."
                });
            }
            catch (Exception ex)
            {
                // TEMPORARY DEBUGGING: expose the actual error.
                return StatusCode(500, new
                {
                    success = false,
                    message = ex.Message,
                    detail = ex.InnerException?.Message
                });
            }
        }

        [HttpGet]
        public IActionResult Index()
        {
            if (HttpContext.Session.GetInt32("UserId") == null)
            {
                return RedirectToAction("Login", "Account");
            }

            HttpContext.Session.Remove("LastAnalyzedMediaHash");

            return View(new MediaUploadModel());
        }

        [HttpPost]
        [RequestSizeLimit(100_000_000)]
        public async Task<IActionResult> Index(MediaUploadModel model)
        {
            int? userId = HttpContext.Session.GetInt32("UserId");

            if (userId == null)
            {
                return RedirectToAction("Login", "Account");
            }

            HttpContext.Session.Remove("LastAnalyzedMediaHash");

            if (model.MediaFile == null || model.MediaFile.Length == 0)
            {
                ViewBag.Error =
                    "Please select an image, video, or audio file.";

                return View(model);
            }

            string extension =
                Path.GetExtension(model.MediaFile.FileName);

            bool isImage = ImageExtensions.Contains(extension);
            bool isVideo = VideoExtensions.Contains(extension);
            bool isAudio = AudioExtensions.Contains(extension);

            if (!isImage && !isVideo && !isAudio)
            {
                ViewBag.Error =
                    "Unsupported file type. Upload JPG, JPEG, PNG, MP4, MOV, AVI, MKV, WEBM, WAV, MP3, FLAC, OGG, or M4A.";

                return View(model);
            }

            string uploadsFolder = Path.Combine(
                _environment.WebRootPath,
                "uploads");

            Directory.CreateDirectory(uploadsFolder);

            string fileName =
                Guid.NewGuid() + extension.ToLowerInvariant();

            string filePath = Path.Combine(uploadsFolder, fileName);
            string relativePath = "/uploads/" + fileName;

            try
            {
                using (var stream = new FileStream(
                    filePath, FileMode.Create))
                {
                    await model.MediaFile.CopyToAsync(stream);
                }

                using var client = new HttpClient
                {
                    Timeout = TimeSpan.FromMinutes(5)
                };

                using var form = new MultipartFormDataContent();

                using var fileStream =
                    System.IO.File.OpenRead(filePath);

                var fileContent = new StreamContent(fileStream);

                string contentType;

                if (isVideo)
                {
                    contentType =
                        "video/" +
                        extension.TrimStart('.').ToLowerInvariant();
                }
                else if (isAudio)
                {
                    contentType = extension.ToLowerInvariant() switch
                    {
                        ".wav" => "audio/wav",
                        ".mp3" => "audio/mpeg",
                        ".flac" => "audio/flac",
                        ".ogg" => "audio/ogg",
                        ".m4a" => "audio/mp4",
                        _ => "application/octet-stream"
                    };
                }
                else
                {
                    contentType = string.IsNullOrWhiteSpace(
                        model.MediaFile.ContentType)
                        ? "application/octet-stream"
                        : model.MediaFile.ContentType;
                }

                fileContent.Headers.ContentType =
                    new MediaTypeHeaderValue(contentType);

                form.Add(fileContent, "file", fileName);

                string endpoint;

                if (isAudio)
                {
                    endpoint =
                        "http://127.0.0.1:5002/predict";
                }
                else if (isVideo)
                {
                    endpoint =
                        "http://127.0.0.1:5001/predict_video";
                }
                else
                {
                    endpoint =
                        "http://127.0.0.1:5001/predict";
                }

                var response =
                    await client.PostAsync(endpoint, form);

                var json =
                    await response.Content.ReadAsStringAsync();

                if (!response.IsSuccessStatusCode)
                {
                    ViewBag.Error =
                        "The AI server could not analyze this file. " +
                        "Details: " + json;

                    return View(model);
                }

                using var document = JsonDocument.Parse(json);
                var root = document.RootElement;

                bool success = root.TryGetProperty(
                    "success", out var successElement
                ) && successElement.ValueKind == JsonValueKind.True;

                if (!success)
                {
                    string apiMessage =
                        root.TryGetProperty("message", out var errorElement)
                            ? errorElement.GetString() ?? "Unknown error."
                            : "The AI server did not return a successful result.";

                    ViewBag.Error = apiMessage;

                    return View(model);
                }

                string prediction =
                    root.TryGetProperty(
                        "prediction", out var predictionElement)
                        ? predictionElement.GetString() ?? "UNKNOWN"
                        : "UNKNOWN";

                double confidence = 0;
                double rawPrediction = 0;

                if (isAudio)
                {
                    double realScore =
                        root.TryGetProperty(
                            "real_score", out var realScoreElement)
                            ? realScoreElement.GetDouble()
                            : 0;

                    double fakeScore =
                        root.TryGetProperty(
                            "fake_score", out var fakeScoreElement)
                            ? fakeScoreElement.GetDouble()
                            : 0;

                    if (prediction.Equals(
                        "REAL",
                        StringComparison.OrdinalIgnoreCase))
                    {
                        confidence = realScore;
                    }
                    else if (prediction.Equals(
                        "FAKE",
                        StringComparison.OrdinalIgnoreCase))
                    {
                        confidence = fakeScore;
                    }
                    else
                    {
                        ViewBag.Error =
                            "The audio AI returned an unknown prediction.";

                        return View(model);
                    }

                    ViewBag.RealScore = realScore;
                    ViewBag.FakeScore = fakeScore;

                    ViewBag.AudioDuration =
                        root.TryGetProperty(
                            "duration_seconds", out var durationElement)
                            ? durationElement.GetDouble()
                            : 0;

                    ViewBag.SampleRate =
                        root.TryGetProperty(
                            "sample_rate_hz", out var sampleRateElement)
                            ? sampleRateElement.GetInt32()
                            : 16000;

                    ViewBag.AudioWarning =
                        root.TryGetProperty(
                            "warning", out var warningElement)
                            ? warningElement.GetString()
                            : "Model scores are not calibrated probabilities.";
                }
                else
                {
                    confidence =
                        root.TryGetProperty(
                            "confidence", out var confidenceElement)
                            ? confidenceElement.GetDouble()
                            : 0;

                    rawPrediction =
                        root.TryGetProperty(
                            "raw_prediction", out var rawElement)
                            ? rawElement.GetDouble()
                            : 0;
                }

                // Generate the SHA-256 fingerprint.
                string? mediaHash = null;

                try
                {
                    using (var mediaStream =
                        System.IO.File.OpenRead(filePath))
                    {
                        mediaHash =
                            _mediaHashService.ComputeSha256(mediaStream);
                    }

                    ViewBag.MediaHash = mediaHash;

                    HttpContext.Session.SetString(
                        "LastAnalyzedMediaHash",
                        mediaHash);

                    // Verify the fingerprint on Ethereum.
                    var blockchainRecord =
                        await _blockchainService.VerifyMediaAsync(
                            mediaHash);

                    ViewBag.BlockchainChecked = true;
                    ViewBag.BlockchainRegistered =
                        blockchainRecord.Exists;

                    if (blockchainRecord.Exists)
                    {
                        ViewBag.BlockchainRegisteredBy =
                            blockchainRecord.RegisteredBy;

                        ViewBag.BlockchainRegisteredAt =
                            DateTimeOffset
                                .FromUnixTimeSeconds(
                                    (long)blockchainRecord.RegisteredAt)
                                .UtcDateTime;
                    }
                }
                catch (Exception)
                {
                    ViewBag.BlockchainChecked = false;

                    ViewBag.BlockchainError =
                        "Blockchain verification is currently unavailable. " +
                        "Your AI detection result is still available.";
                }

                // Save the AI detection and media fingerprint in the database.
                var history = new DetectionHistory
                {
                    UserId = userId.Value,
                    FileName = fileName,
                    FilePath = relativePath,
                    Prediction = prediction,
                    Confidence = confidence,
                    RawPrediction = rawPrediction,
                    DateAnalyzed = DateTime.Now,
                    MediaHash = mediaHash
                };

                // If the fingerprint is already registered, synchronize its status.
                if (ViewBag.BlockchainRegistered == true)
                {
                    history.BlockchainRegistered = true;
                    history.BlockchainRegisteredBy =
                        ViewBag.BlockchainRegisteredBy;
                    history.BlockchainRegisteredAtUtc =
                        ViewBag.BlockchainRegisteredAt;
                }

                _context.DetectionHistories.Add(history);

                await _context.SaveChangesAsync();

                ViewBag.FileName = fileName;
                ViewBag.FilePath = relativePath;
                ViewBag.Prediction = prediction;
                ViewBag.Confidence = confidence;
                ViewBag.IsVideo = isVideo;
                ViewBag.IsAudio = isAudio;
                ViewBag.IsImage = isImage;

                ViewBag.Message =
                    root.TryGetProperty(
                        "message", out var messageElement)
                        ? messageElement.GetString()
                        : null;

                if (isVideo)
                {
                    ViewBag.FramesAnalyzed =
                        root.TryGetProperty(
                            "frames_analyzed", out var framesElement)
                            ? framesElement.GetInt32()
                            : 0;

                    ViewBag.FrameResults =
                        root.TryGetProperty(
                            "frame_results", out var frames)
                            ? frames.GetRawText()
                            : "[]";
                }
            }
            catch (TaskCanceledException)
            {
                ViewBag.Error =
                    "The AI request timed out. Try a shorter or smaller file.";
            }
            catch (HttpRequestException ex)
            {
                ViewBag.Error =
                    "Could not connect to the AI server. " +
                    "Make sure the required detection API is running. " +
                    ex.Message;
            }
            catch (Exception ex)
            {
                ViewBag.Error =
                    "Unable to complete the analysis: " + ex.Message;
            }

            return View(model);
        }
    }
}
