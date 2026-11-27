namespace rAIdar.Models
{
    public class PredictionResult
    {
        public string Prediction { get; set; } = "";

        public double Confidence { get; set; }

        public double RawPrediction { get; set; }
    }
}