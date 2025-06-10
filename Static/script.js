const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const uploadForm = document.getElementById("uploadForm");
const output = document.getElementById("output");
const submitBtn = document.getElementById("submitBtn");

const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/gif'];

// Handle image preview and type check
imageInput.addEventListener("change", function () {
  const file = imageInput.files[0];

  if (!file) return;

  if (!allowedTypes.includes(file.type)) {
    alert("❌ Invalid file type. Please upload an image file (jpg, png, gif).");
    imageInput.value = "";
    imagePreview.style.display = "none";
    output.innerText = "Upload an image to get started";
    return;
  }

  imagePreview.src = URL.createObjectURL(file);
  imagePreview.style.display = "block";
  output.innerText = "✅ Image loaded. Ready to classify.";
});

// Prevent form submit if no image
submitBtn.addEventListener("click", function (e) {
  if (!imageInput.files[0]) {
    e.preventDefault();
    alert("⚠️ Please upload a photo before clicking 'Classify Image'.");
  }
});

// Handle form submission and image classification
uploadForm.addEventListener("submit", async function (e) {
  e.preventDefault();

  const file = imageInput.files[0];

  if (!file) {
    output.textContent = '⚠️ Please select an image first.';
    return;
  }

  if (!allowedTypes.includes(file.type)) {
    alert("❌ The file you uploaded is not a valid image. Please upload JPG, PNG, or GIF.");
    imageInput.value = "";
    imagePreview.style.display = "none";
    output.innerText = "Upload an image to get started";
    return;
  }

  const formData = new FormData();
  formData.append('image', file);

  output.textContent = '⏳ Processing...';

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    const predictions = await response.json();

    if (response.ok) {
      output.innerHTML = predictions.map(p =>
        `<p><strong>${p.label}</strong>: ${(p.probability * 100).toFixed(2)}%</p>`
      ).join('');
    } else {
      output.textContent = predictions.error || '❌ Something went wrong.';
    }
  } catch (err) {
    output.textContent = '❌ Error connecting to server.';
  }
});
