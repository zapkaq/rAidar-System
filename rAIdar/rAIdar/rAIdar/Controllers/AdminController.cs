using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using rAIdar.Data;

namespace rAIdar.Controllers
{
    public class AdminController : Controller
    {
        private readonly ApplicationDbContext _context;

        public AdminController(ApplicationDbContext context)
        {
            _context = context;
        }

        public async Task<IActionResult> Dashboard()
        {
            ViewBag.TotalUsers = await _context.AppUsers.CountAsync();
            ViewBag.ActiveUsers = await _context.AppUsers.CountAsync(u => u.IsActive);
            ViewBag.TotalScans = await _context.DetectionHistories.CountAsync();
            ViewBag.RealImages = await _context.DetectionHistories.CountAsync(h => h.Prediction == "REAL");
            ViewBag.FakeImages = await _context.DetectionHistories.CountAsync(h => h.Prediction == "FAKE");

            var recentDetections = await _context.DetectionHistories
                .Include(h => h.User)
                .OrderByDescending(h => h.DateAnalyzed)
                .Take(10)
                .ToListAsync();

            return View(recentDetections);
        }

        public async Task<IActionResult> Users()
        {
            var users = await _context.AppUsers
                .Include(u => u.DetectionHistories)
                .OrderByDescending(u => u.CreatedAt)
                .ToListAsync();

            return View(users);
        }

        [HttpPost]
        public async Task<IActionResult> ToggleUserStatus(int id)
        {
            var user = await _context.AppUsers.FindAsync(id);

            if (user != null)
            {
                user.IsActive = !user.IsActive;
                await _context.SaveChangesAsync();
            }

            return RedirectToAction("Users");
        }

        public async Task<IActionResult> Reports(string search, string prediction)
        {
            var query = _context.DetectionHistories
                .Include(h => h.User)
                .AsQueryable();

            if (!string.IsNullOrWhiteSpace(search))
            {
                query = query.Where(h =>
                    h.FileName.Contains(search) ||
                    h.User!.Username.Contains(search) ||
                    h.User.Email.Contains(search));
            }

            if (!string.IsNullOrWhiteSpace(prediction) && prediction != "ALL")
            {
                query = query.Where(h => h.Prediction == prediction);
            }

            ViewBag.Search = search;
            ViewBag.SelectedPrediction = prediction;

            ViewBag.TotalScans = await query.CountAsync();
            ViewBag.RealImages = await query.CountAsync(h => h.Prediction == "REAL");
            ViewBag.FakeImages = await query.CountAsync(h => h.Prediction == "FAKE");

            var reports = await query
                .OrderByDescending(h => h.DateAnalyzed)
                .ToListAsync();

            return View(reports);
        }

        public IActionResult Settings()
        {
            return View();
        }
    }
}