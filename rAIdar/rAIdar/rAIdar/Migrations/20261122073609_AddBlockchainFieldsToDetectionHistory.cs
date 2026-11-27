using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace rAIdar.Migrations
{
    /// <inheritdoc />
    public partial class AddBlockchainFieldsToDetectionHistory : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<bool>(
                name: "BlockchainRegistered",
                table: "DetectionHistories",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<DateTime>(
                name: "BlockchainRegisteredAtUtc",
                table: "DetectionHistories",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "BlockchainRegisteredBy",
                table: "DetectionHistories",
                type: "nvarchar(42)",
                maxLength: 42,
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "BlockchainTransactionHash",
                table: "DetectionHistories",
                type: "nvarchar(66)",
                maxLength: 66,
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "MediaHash",
                table: "DetectionHistories",
                type: "nvarchar(64)",
                maxLength: 64,
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "BlockchainRegistered",
                table: "DetectionHistories");

            migrationBuilder.DropColumn(
                name: "BlockchainRegisteredAtUtc",
                table: "DetectionHistories");

            migrationBuilder.DropColumn(
                name: "BlockchainRegisteredBy",
                table: "DetectionHistories");

            migrationBuilder.DropColumn(
                name: "BlockchainTransactionHash",
                table: "DetectionHistories");

            migrationBuilder.DropColumn(
                name: "MediaHash",
                table: "DetectionHistories");
        }
    }
}
