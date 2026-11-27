const form = document.querySelector("form");
const input = document.getElementById("mediaInput");

if (input) {

    input.addEventListener("change", function () {

        const file = this.files[0];

        if (!file) return;

        document.getElementById("imagePreview").style.display = "none";
        document.getElementById("videoPreview").style.display = "none";
        document.getElementById("audioPreview").style.display = "none";

        const url = URL.createObjectURL(file);

        if (file.type.startsWith("image")) {

            const img = document.getElementById("imagePreview");

            img.src = url;

            img.style.display = "block";
        }

        else if (file.type.startsWith("video")) {

            const video = document.getElementById("videoPreview");

            video.src = url;

            video.style.display = "block";
        }

        else if (file.type.startsWith("audio")) {

            const audio = document.getElementById("audioPreview");

            audio.src = url;

            audio.style.display = "block";
        }

    });

}

if (form) {

    form.addEventListener("submit", function () {

        document.getElementById("btnText").style.display = "none";
        document.getElementById("loadingSpinner").style.display = "inline-block";

    });

}