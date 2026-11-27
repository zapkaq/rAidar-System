
import { network } from "hardhat";

async function main() {
    const { ethers } = await network.create();

    const networkInfo = await ethers.provider.getNetwork();
    const [signer] = await ethers.getSigners();

    const balance = await ethers.provider.getBalance(signer.address);

    console.log("Connected to Ethereum network!");
    console.log("Chain ID:", networkInfo.chainId.toString());
    console.log(
        "Wallet balance:",
        ethers.formatEther(balance),
        "SepoliaETH"
    );
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
