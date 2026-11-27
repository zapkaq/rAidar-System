
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using rAIdar.Data;
using rAIdar.Models;
using System.Security.Cryptography;
using System.Text;

namespace rAIdar.Controllers
{
    public class AccountController : Controller
    {
        private readonly ApplicationDbContext _context;

        public AccountController(ApplicationDbContext context)
        {
            _context = context;
        }

        // GET: Account/Login
        [HttpGet]
        public IActionResult Login()
        {
            if (HttpContext.Session.GetInt32("UserId") != null)
            {
                return RedirectToAction("Index", "Dashboard");
            }

            return View();
        }

        // GET: Account/Register
        [HttpGet]
        public IActionResult Register()
        {
            if (HttpContext.Session.GetInt32("UserId") != null)
            {
                return RedirectToAction("Index", "Dashboard");
            }

            return View();
        }

        // POST: Account/Register
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Register(RegisterViewModel model)
        {
            if (HttpContext.Session.GetInt32("UserId") != null)
            {
                return RedirectToAction("Index", "Dashboard");
            }

            if (!ModelState.IsValid)
            {
                return View(model);
            }

            bool userExists = await _context.AppUsers.AnyAsync(u =>
                u.Username == model.Username ||
                u.Email == model.Email);

            if (userExists)
            {
                ModelState.AddModelError(
                    string.Empty,
                    "Username or email already exists."
                );

                return View(model);
            }

            var user = new AppUser
            {
                FullName = model.FullName,
                Username = model.Username,
                Email = model.Email,
                PasswordHash = HashPassword(model.Password),
                Role = "User",
                IsActive = true,
                CreatedAt = DateTime.Now
            };

            _context.AppUsers.Add(user);

            await _context.SaveChangesAsync();

            TempData["Success"] =
                "Registration successful. You may now log in.";

            return RedirectToAction("Login");
        }

        // POST: Account/Login
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Login(string username, string password)
        {
            if (HttpContext.Session.GetInt32("UserId") != null)
            {
                return RedirectToAction("Index", "Dashboard");
            }

            if (string.IsNullOrWhiteSpace(username) ||
                string.IsNullOrWhiteSpace(password))
            {
                ModelState.AddModelError(
                    string.Empty,
                    "Please enter your username and password."
                );

                return View();
            }

            string hashedPassword = HashPassword(password);

            var user = await _context.AppUsers.FirstOrDefaultAsync(u =>
                u.Username == username &&
                u.PasswordHash == hashedPassword &&
                u.IsActive);

            if (user == null)
            {
                ModelState.AddModelError(
                    string.Empty,
                    "Invalid username or password."
                );

                return View();
            }

            // Create the user session.
            HttpContext.Session.SetInt32("UserId", user.UserId);
            HttpContext.Session.SetString("Username", user.Username);
            HttpContext.Session.SetString("Role", user.Role);

            TempData["Success"] = "Login successful.";

            if (string.Equals(
                user.Role,
                "Admin",
                StringComparison.OrdinalIgnoreCase))
            {
                return RedirectToAction("Dashboard", "Admin");
            }

            return RedirectToAction("Index", "Dashboard");
        }

        // POST: Account/Logout
        [HttpPost]
        [ValidateAntiForgeryToken]
        public IActionResult Logout()
        {
            HttpContext.Session.Clear();

            TempData["Success"] = "You have successfully logged out.";

            return RedirectToAction("Login");
        }

        // Existing password hashing method.
        // This is retained temporarily to avoid breaking existing accounts.
        private string HashPassword(string password)
        {
            using var sha256 = SHA256.Create();

            byte[] bytes = sha256.ComputeHash(
                Encoding.UTF8.GetBytes(password)
            );

            return Convert.ToBase64String(bytes);
        }
    }
}
