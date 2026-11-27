using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using rAIdar.Data;

namespace rAIdar.Controllers
{
    public class ProfileController : Controller
    {
        private readonly ApplicationDbContext _context;

        public ProfileController(ApplicationDbContext context)
        {
            _context = context;
        }

        public async Task<IActionResult> Index()
        {
            int? userId = HttpContext.Session.GetInt32("UserId");

            if (userId == null)
                return RedirectToAction("Login", "Account");

            var user = await _context.AppUsers
                .FirstOrDefaultAsync(u => u.UserId == userId.Value);

            if (user == null)
                return RedirectToAction("Login", "Account");

            ViewBag.TotalDetections = await _context.DetectionHistories
                .CountAsync(h => h.UserId == userId.Value);

            ViewBag.RealImages = await _context.DetectionHistories
                .CountAsync(h => h.UserId == userId.Value && h.Prediction == "REAL");

            ViewBag.FakeImages = await _context.DetectionHistories
                .CountAsync(h => h.UserId == userId.Value && h.Prediction == "FAKE");

            return View(user);
        }
    }
}