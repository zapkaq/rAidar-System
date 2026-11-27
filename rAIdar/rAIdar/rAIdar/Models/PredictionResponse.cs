namespace rAIdar.Models
{
    public class PredictionResponse
    {
        public string Prediction { get; set; } = "";

        public double Confidence { get; set; }

        public string Filename { get; set; } = "";
    }
}