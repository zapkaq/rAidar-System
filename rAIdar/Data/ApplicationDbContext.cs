using Microsoft.EntityFrameworkCore;
using rAIdar.Models;

namespace rAIdar.Data
{
    public class ApplicationDbContext : DbContext
    {
        public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options)
            : base(options)
        {
        }

        public DbSet<AppUser> AppUsers { get; set; }
        public DbSet<DetectionHistory> DetectionHistories { get; set; }
    }
}
