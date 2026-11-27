using Microsoft.AspNetCore.Http;
using System.ComponentModel.DataAnnotations;

namespace rAIdar.Models
{
    public class MediaUploadModel
    {
        [Required]
        public IFormFile? MediaFile { get; set; }
    }
}