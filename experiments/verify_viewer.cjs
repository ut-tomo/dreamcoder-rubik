const fs=require('fs');
const src=fs.readFileSync('docs/app.js','utf8');
const D=JSON.parse(fs.readFileSync('docs/data.json'));
const model=src.split('// ---------------------------------------------------------------- 3D view')[0].replace(/^import .*$/m,'');
const M=new Function('D',model+`STICKERS=D.stickers;STICKERS.forEach(([p,n],i)=>INDEX.set(key(p,n),i));buildRotations();return {applyMove,applyMatrix,normalizingRotation};`)(D);
let n=0;
for(const r of D.runs) for(const a of Object.values(r.arms)) for(const t of r.tasks.filter(t=>t.split==='test')) {
 const sol=a.solutions[t.name]; if(!sol)continue;
 let s=t.start;
 sol.moves.forEach((m,i)=>{s=M.applyMove(s,m);if((sol.boundaries||[]).includes(i+1))s=M.applyMatrix(s,M.normalizingRotation(s));});
 s=M.applyMatrix(s,M.normalizingRotation(s));
 if(!s.every((v,i)=>v===i))throw new Error(r.id+' '+t.name);
 n++;
}
console.log('JavaScript viewer model replay passed:',n,'full-cube solutions across 24 runs');
