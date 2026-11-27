
import { buildModule } from "@nomicfoundation/hardhat-ignition/modules";

export default buildModule("MediaVerificationModule", (m) => {

    const mediaVerification = m.contract("MediaVerification");

    return {
        mediaVerification
    };

});
