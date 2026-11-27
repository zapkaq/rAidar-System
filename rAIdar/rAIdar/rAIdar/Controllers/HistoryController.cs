using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using rAIdar.Data;

namespace rAIdar.Controllers
{
    public class HistoryController : Controller
    {
        private readonly ApplicationDbContext _context;

        public HistoryController(ApplicationDbContext context)
        {
            _context = context;
        }

        public async Task<IActionResult> Index(string search, string prediction)
        {
            int? userId = HttpContext.Session.GetInt32("UserId");

            if (userId == null)
                return RedirectToAction("Login", "Account");

            var query = _context.DetectionHistories
                .Where(h => h.UserId == userId.Value)
                .AsQueryable();

            if (!string.IsNullOrWhiteSpace(search))
            {
                query = query.Where(h => h.FileName.Contains(search));
            }

            if (!string.IsNullOrWhiteSpace(prediction) && prediction != "ALL")
            {
                query = query.Where(h => h.Prediction == prediction);
            }

            ViewBag.Search = search;
            ViewBag.SelectedPrediction = prediction;

            var histories = await query
                .OrderByDescending(h => h.DateAnalyzed)
                .ToListAsync();

            return View(histories);
        }
    }
}