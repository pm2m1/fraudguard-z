
const $=id=>document.getElementById(id);
const td=(row,value)=>{const el=document.createElement('td');el.textContent=String(value);row.appendChild(el)};
async function getJSON(url,opts){const response=await fetch(url,{cache:'no-store',...opts});const body=await response.json();if(!response.ok)throw Error(body.error||('HTTP '+response.status));return body}
function renderEvidence(container,data){
  container.textContent=JSON.stringify(data,null,2);
  const list=$('evidence-items');
  list.replaceChildren();
  const observed=data.evidence && Array.isArray(data.evidence.items) ? data.evidence.items : [];
  for(const item of observed){
    const el=document.createElement('div');el.className='evidence-item';
    const name=document.createElement('strong');name.textContent=String(item.code||'Observed context');
    const value=document.createElement('span');value.textContent=String(item.observed)+' '+String(item.unit||'');
    const description=document.createElement('p');description.textContent=String(item.description||'');
    el.append(name,value,description);list.appendChild(el);
  }
}
async function refresh(){try{
const [s,h,recent]=await Promise.all([getJSON('/stats'),getJSON('/health'),getJSON('/transactions/recent?limit=20')]);
$('total').textContent=s.total;$('allowed').textContent=s.allowed;$('reviewed').textContent=s.reviewed;
$('health-indicator').textContent=h.healthy?'Healthy · local simulator':'Unavailable';
$('health-indicator').className='status '+(h.healthy?'ok':'error');
$('metadata').textContent='Model '+h.model_version+' · Policy '+h.policy_version+' · '+h.scoring_source+' · healthy='+h.healthy;
$('rows').replaceChildren();for(const item of recent.transactions){const row=document.createElement('tr');td(row,item.event_id);td(row,item.step);td(row,item.type);td(row,item.amount);td(row,Number(item.risk).toFixed(6));const decisionCell=document.createElement('td');
const tag=document.createElement('span');tag.className='table-tag '+(item.decision==='REVIEW'?'review':'allow');tag.textContent=item.decision;
decisionCell.appendChild(tag);row.appendChild(decisionCell);
td(row,item.recipient_prior_originators);const cell=document.createElement('td');const b=document.createElement('button');b.textContent='View';b.addEventListener('click',async()=>{try{const d=await getJSON('/transactions/'+encodeURIComponent(item.event_id)+'/evidence');renderEvidence($('result'),d);$('message').textContent='Stored observation for '+item.event_id}catch(e){$('message').textContent=e.message}});cell.appendChild(b);row.appendChild(cell);$('rows').appendChild(row)}
}catch(e){
$('metadata').textContent='Unavailable: '+e.message;$('metadata').className='alert';
$('health-indicator').textContent='Unavailable · no new decisions';
$('health-indicator').className='status error';
$('total').textContent='—';$('allowed').textContent='—';$('reviewed').textContent='—';
$('rows').replaceChildren();
}}
$('refresh-now').addEventListener('click',refresh);
$('form').addEventListener('submit',async e=>{e.preventDefault();const submit=$('submit-score');submit.disabled=true;const data=Object.fromEntries(new FormData(e.currentTarget));for(const key of ['step','source_row_index','amount','oldbalanceOrg'])data[key]=Number(data[key]);const started=performance.now();try{const r=await getJSON('/transactions/score',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});$('message').textContent=r.decision+' · '+(r.duplicate?'Duplicate':'New commit');renderEvidence($('result'),r);await refresh()}catch(error){$('message').textContent='Error: '+error.message}finally{submit.disabled=false;$('latency').textContent='Browser request + UI refresh: '+(performance.now()-started).toFixed(1)+' ms (not inference latency)'}});
refresh();
