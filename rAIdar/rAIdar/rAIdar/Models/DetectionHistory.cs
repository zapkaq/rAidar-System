
using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace rAIdar.Models
{
    public class DetectionHistory
    {
        [Key]
        public int DetectionId { get; set; }

        [Required]
        public string FileName { get; set; } = "";

        [Required]
        public string FilePath { get; set; } = "";

        [Required]
        public string Prediction { get; set; } = "";

        public double Confidence { get; set; }

        public double RawPrediction { get; set; }

        public DateTime DateAnalyzed { get; set; } = DateTime.Now;

        // Foreign Key
        public int UserId { get; set; }

        [ForeignKey("UserId")]
        public AppUser? User { get; set; }

        // Blockchain Verification Information

        [MaxLength(64)]
        public string? MediaHash { get; set; }

        public bool BlockchainRegistered { get; set; } = false;

        [MaxLength(66)]
        public string? BlockchainTransactionHash { get; set; }

        [MaxLength(42)]
        public string? BlockchainRegisteredBy { get; set; }

        public DateTime? BlockchainRegisteredAtUtc { get; set; }
    }
}
