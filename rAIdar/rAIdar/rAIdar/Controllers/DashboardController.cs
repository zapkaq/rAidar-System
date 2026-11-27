using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using rAIdar.Data;

namespace rAIdar.Controllers
{
    public class DashboardController : Controller
    {
        private readonly ApplicationDbContext _context;

        public DashboardController(ApplicationDbContext context)
        {
            _context = context;
        }

        public async Task<IActionResult> Index()
        {
            int? userId = HttpContext.Session.GetInt32("UserId");

            if (userId == null)
                return RedirectToAction("Login", "Account");

            var records = _context.DetectionHistories
                .Where(h => h.UserId == userId.Value);

            ViewBag.TotalScans = await records.CountAsync();
            ViewBag.RealImages = await records.CountAsync(h => h.Prediction == "REAL");
            ViewBag.FakeImages = await records.CountAsync(h => h.Prediction == "FAKE");

            ViewBag.RecentDetections = await records
                .OrderByDescending(h => h.DateAnalyzed)
                .Take(5)
                .ToListAsync();

            return View();
        }
    }
}