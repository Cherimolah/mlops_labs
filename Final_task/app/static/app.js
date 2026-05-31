document.getElementById('predict-form').addEventListener('submit', async function (e) {
  e.preventDefault();
  const btn = document.getElementById('submit-btn');
  const spinner = document.getElementById('spinner');
  const resultBox = document.getElementById('result-box');
  const errorBox = document.getElementById('error-box');
  const priceEl = document.getElementById('result-price');

  resultBox.style.display = 'none';
  errorBox.style.display = 'none';
  btn.disabled = true;
  spinner.style.display = 'inline-block';

  const fd = new FormData(this);
  const data = {
    full_sq: parseFloat(fd.get('full_sq')),
    life_sq: parseFloat(fd.get('life_sq')),
    floor: parseInt(fd.get('floor')),
    max_floor: parseInt(fd.get('max_floor')),
    build_year: parseInt(fd.get('build_year')),
    num_room: parseInt(fd.get('num_room')),
    kitch_sq: parseFloat(fd.get('kitch_sq')),
    sub_area: fd.get('sub_area'),
  };

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    const json = await resp.json();
    if (!resp.ok) {
      errorBox.textContent = json.detail || 'Ошибка сервера';
      errorBox.style.display = 'block';
    } else {
      priceEl.textContent = json.price_formatted;
      resultBox.style.display = 'block';
      setTimeout(() => location.reload(), 800);
    }
  } catch (err) {
    errorBox.textContent = 'Не удалось подключиться к серверу';
    errorBox.style.display = 'block';
  } finally {
    btn.disabled = false;
    spinner.style.display = 'none';
  }
});