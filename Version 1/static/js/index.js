async function predict() {
  const symptomText = document.getElementById("symptom").value;
  const resultDiv = document.getElementById("result");
  const nextStepDiv = document.getElementById("next-step");

  resultDiv.textContent = "جارٍ التشخيص...";
  nextStepDiv.innerHTML = "";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: symptomText })
    });

    const data = await response.json();

    if (response.ok) {
      resultDiv.innerHTML = `التخصص الطبي المناسب هو: <strong>${data.specialty}</strong>`;
      nextStepDiv.innerHTML = `
        <button onclick="goToSpecialty('${data.specialty}')">روح اكشف عند دكتور : ${data.specialty}</button>
      `;
    } else {
      resultDiv.textContent = data.error || "حدث خطأ أثناء التشخيص.";
    }
  } catch (error) {
    resultDiv.textContent = "فشل الاتصال بالخادم.";
  }
}

function goToSpecialty(specialty) {
  window.location.href = `/specialty/${encodeURIComponent(specialty)}`;
}
