
using System;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Nethereum.ABI.FunctionEncoding.Attributes;
using Nethereum.Contracts;
using Nethereum.Hex.HexTypes;
using Nethereum.Web3;
using Nethereum.Web3.Accounts;
using Nethereum.RPC.Eth.DTOs;

namespace rAIdar.Services
{
    public class BlockchainService
    {
        private readonly Web3 _web3;
        private readonly Account _account;
        private readonly string _contractAddress;

        private const int LogScanBatchSize = 1000;

        private const string ContractAbi = @"[
            {
                ""inputs"": [
                    {
                        ""internalType"": ""bytes32"",
                        ""name"": ""mediaHash"",
                        ""type"": ""bytes32""
                    }
                ],
                ""name"": ""registerMedia"",
                ""outputs"": [],
                ""stateMutability"": ""nonpayable"",
                ""type"": ""function""
            },
            {
                ""inputs"": [
                    {
                        ""internalType"": ""bytes32"",
                        ""name"": ""mediaHash"",
                        ""type"": ""bytes32""
                    }
                ],
                ""name"": ""verifyMedia"",
                ""outputs"": [
                    {
                        ""internalType"": ""bool"",
                        ""name"": ""exists"",
                        ""type"": ""bool""
                    },
                    {
                        ""internalType"": ""address"",
                        ""name"": ""registeredBy"",
                        ""type"": ""address""
                    },
                    {
                        ""internalType"": ""uint256"",
                        ""name"": ""registeredAt"",
                        ""type"": ""uint256""
                    }
                ],
                ""stateMutability"": ""view"",
                ""type"": ""function""
            }
        ]";

        public BlockchainService(IConfiguration configuration)
        {
            string rpcUrl =
                configuration["Blockchain:RpcUrl"]
                ?? throw new InvalidOperationException(
                    "Blockchain RPC URL is missing."
                );

            string privateKey =
                configuration["Blockchain:PrivateKey"]
                ?? throw new InvalidOperationException(
                    "Blockchain private key is missing."
                );

            _contractAddress =
                configuration["Blockchain:ContractAddress"]
                ?? throw new InvalidOperationException(
                    "Contract address is missing."
                );

            if (!long.TryParse(
                configuration["Blockchain:ChainId"],
                out long chainId))
            {
                throw new InvalidOperationException(
                    "Invalid blockchain Chain ID."
                );
            }

            _account = new Account(privateKey, chainId);
            _web3 = new Web3(_account, rpcUrl);
        }

        // Register a SHA-256 fingerprint on Ethereum Sepolia.
        public async Task<string> RegisterMediaAsync(string sha256Hash)
        {
            byte[] hashBytes = ConvertHash(sha256Hash);

            var contract = _web3.Eth.GetContract(
                ContractAbi,
                _contractAddress
            );

            var function = contract.GetFunction("registerMedia");

            var gasEstimate = await function.EstimateGasAsync(
                _account.Address,
                null,
                null,
                hashBytes
            );

            BigInteger estimatedGas = gasEstimate.Value;
            BigInteger adjustedGas = estimatedGas * 120 / 100;

            var gasLimit = new HexBigInteger(adjustedGas);

            string transactionHash = await function.SendTransactionAsync(
                _account.Address,
                gasLimit,
                null,
                hashBytes
            );

            DateTimeOffset timeout =
                DateTimeOffset.UtcNow.AddMinutes(2);

            TransactionReceipt? receipt = null;

            while (receipt == null)
            {
                if (DateTimeOffset.UtcNow >= timeout)
                {
                    throw new TimeoutException(
                        "Ethereum transaction confirmation timed out. " +
                        "Check the transaction on Etherscan: " +
                        "https://sepolia.etherscan.io/tx/" + transactionHash
                    );
                }

                await Task.Delay(2000);

                receipt = await _web3.Eth.Transactions
                    .GetTransactionReceipt
                    .SendRequestAsync(transactionHash);
            }

            if (receipt.Status == null ||
                receipt.Status.Value != BigInteger.One)
            {
                throw new InvalidOperationException(
                    "The Ethereum transaction was mined but failed. " +
                    "Transaction: " + transactionHash
                );
            }

            return transactionHash;
        }

        // Verify a fingerprint against Ethereum.
        public async Task<MediaRecordOutput> VerifyMediaAsync(
            string sha256Hash)
        {
            byte[] hashBytes = ConvertHash(sha256Hash);

            var contract = _web3.Eth.GetContract(
                ContractAbi,
                _contractAddress
            );

            var function = contract.GetFunction("verifyMedia");

            return await function.CallDeserializingToObjectAsync<MediaRecordOutput>(
                hashBytes
            );
        }

        // Find the original transaction hash by scanning Ethereum logs
        // in smaller block ranges instead of querying the entire chain.
        public async Task<string?> FindRegistrationTransactionAsync(
            string sha256Hash)
        {
            byte[] hashBytes = ConvertHash(sha256Hash);

            var mediaRegisteredEvent =
                _web3.Eth.GetEvent<MediaRegisteredEventDTO>(
                    _contractAddress
                );

            // Get the latest confirmed block number.
            var latestBlock =
                await _web3.Eth.Blocks.GetBlockNumber.SendRequestAsync();

            BigInteger latestBlockNumber = latestBlock.Value;

            BigInteger toBlock = latestBlockNumber;

            // Scan backwards in batches of 1,000 blocks.
            while (toBlock >= BigInteger.Zero)
            {
                BigInteger fromBlock =
                    BigInteger.Max(
                        BigInteger.Zero,
                        toBlock - LogScanBatchSize + 1
                    );

                var filter = mediaRegisteredEvent.CreateFilterInput(
                    fromBlock: new BlockParameter(
                        new HexBigInteger(fromBlock)
                    ),
                    toBlock: new BlockParameter(
                        new HexBigInteger(toBlock)
                    ),
                    filterTopic1: new object[] { hashBytes }
                );

                var matchingEvents =
                    await mediaRegisteredEvent.GetAllChangesAsync(filter);

                var matchingEvent = matchingEvents.FirstOrDefault(e =>
                    e.Log != null &&
                    !string.IsNullOrWhiteSpace(e.Log.TransactionHash) &&
                    e.Event.MediaHash != null &&
                    e.Event.MediaHash.SequenceEqual(hashBytes)
                );

                if (matchingEvent != null)
                {
                    return matchingEvent.Log.TransactionHash;
                }

                // Move to the preceding block range.
                toBlock = fromBlock - BigInteger.One;

                // Brief pause between RPC requests.
                await Task.Delay(150);
            }

            return null;
        }

        // Validate and convert SHA-256 hexadecimal fingerprint.
        private byte[] ConvertHash(string sha256Hash)
        {
            if (string.IsNullOrWhiteSpace(sha256Hash))
            {
                throw new ArgumentException(
                    "Media hash cannot be empty."
                );
            }

            if (sha256Hash.Length != 64)
            {
                throw new ArgumentException(
                    "Invalid SHA-256 hash. Expected 64 hexadecimal characters."
                );
            }

            try
            {
                return Convert.FromHexString(sha256Hash);
            }
            catch (FormatException)
            {
                throw new ArgumentException(
                    "Media hash contains invalid hexadecimal characters."
                );
            }
        }
    }

    // Smart contract verification result.
    [FunctionOutput]
    public class MediaRecordOutput : IFunctionOutputDTO
    {
        [Parameter("bool", "exists", 1)]
        public bool Exists { get; set; }

        [Parameter("address", "registeredBy", 2)]
        public string RegisteredBy { get; set; } = string.Empty;

        [Parameter("uint256", "registeredAt", 3)]
        public BigInteger RegisteredAt { get; set; }
    }

    // Smart contract MediaRegistered event.
    [Event("MediaRegistered")]
    public class MediaRegisteredEventDTO : IEventDTO
    {
        [Parameter("bytes32", "mediaHash", 1, true)]
        public byte[] MediaHash { get; set; } = Array.Empty<byte>();

        [Parameter("address", "registeredBy", 2, true)]
        public string RegisteredBy { get; set; } = string.Empty;

        [Parameter("uint256", "registeredAt", 3, false)]
        public BigInteger RegisteredAt { get; set; }
    }
}
