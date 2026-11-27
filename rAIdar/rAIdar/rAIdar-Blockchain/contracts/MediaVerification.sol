
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract MediaVerification {

    struct MediaRecord {
        address registeredBy;
        uint256 registeredAt;
        bool exists;
    }

    // SHA-256 fingerprint => registration record
    mapping(bytes32 => MediaRecord) private mediaRecords;

    event MediaRegistered(
        bytes32 indexed mediaHash,
        address indexed registeredBy,
        uint256 registeredAt
    );

    // Register a media fingerprint
    function registerMedia(bytes32 mediaHash) external {
        require(mediaHash != bytes32(0), "Invalid media hash");
        require(
            !mediaRecords[mediaHash].exists,
            "Media already registered"
        );

        mediaRecords[mediaHash] = MediaRecord({
            registeredBy: msg.sender,
            registeredAt: block.timestamp,
            exists: true
        });

        emit MediaRegistered(
            mediaHash,
            msg.sender,
            block.timestamp
        );
    }

    // Check whether a media fingerprint exists
    function verifyMedia(bytes32 mediaHash)
        external
        view
        returns (
            bool exists,
            address registeredBy,
            uint256 registeredAt
        )
    {
        MediaRecord memory record = mediaRecords[mediaHash];

        return (
            record.exists,
            record.registeredBy,
            record.registeredAt
        );
    }
}
