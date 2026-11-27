
import { expect } from "chai";
import { network } from "hardhat";

const { ethers } = await network.create();

describe("MediaVerification", function () {

    it("Should register a media fingerprint", async function () {
        const [owner] = await ethers.getSigners();

        const contract = await ethers.deployContract("MediaVerification");

        const mediaHash = ethers.sha256(
            ethers.toUtf8Bytes("rAIdar test image")
        );

        await expect(contract.registerMedia(mediaHash))
            .to.emit(contract, "MediaRegistered");

        const [exists, registeredBy, registeredAt] =
            await contract.verifyMedia(mediaHash);

        expect(exists).to.equal(true);
        expect(registeredBy).to.equal(owner.address);
        expect(registeredAt).to.be.greaterThan(0n);
    });

    it("Should reject duplicate media fingerprints", async function () {
        const contract = await ethers.deployContract("MediaVerification");

        const mediaHash = ethers.sha256(
            ethers.toUtf8Bytes("rAIdar duplicate image")
        );

        await contract.registerMedia(mediaHash);

        await expect(
            contract.registerMedia(mediaHash)
        ).to.be.revertedWith("Media already registered");
    });

    it("Should return false for an unregistered fingerprint", async function () {
        const contract = await ethers.deployContract("MediaVerification");

        const mediaHash = ethers.sha256(
            ethers.toUtf8Bytes("unknown media")
        );

        const [exists, registeredBy, registeredAt] =
            await contract.verifyMedia(mediaHash);

        expect(exists).to.equal(false);
        expect(registeredBy).to.equal(ethers.ZeroAddress);
        expect(registeredAt).to.equal(0n);
    });

});