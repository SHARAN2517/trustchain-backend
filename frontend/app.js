const $ = id => document.getElementById(id);

function out(txt) {
  $('output').textContent = typeof txt === 'string' ? txt : JSON.stringify(txt, null, 2);
}

async function get(path) {
  const base = $('backendUrl').value.trim();
  if (!base) return out('Set a backend URL');
  const url = new URL(path, base).toString();
  out(`GET ${url} ...`);
  try {
    const res = await fetch(url, { headers: { 'X-API-Key': 'demo-key' } });
    const text = await res.text();
    try { out(JSON.parse(text)); } catch { out(text); }
  } catch (err) {
    out('Error: ' + err.message);
  }
}

$('btnProofs').addEventListener('click', () => get('/blockchain/proofs'));
$('btnVersion').addEventListener('click', () => get('/model/version'));
