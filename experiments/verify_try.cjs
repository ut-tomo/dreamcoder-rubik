// Check the "Try it" solver: random scrambles (all 6 faces), each config's pooled programs, final state must be solved.
const fs=require('fs');
const src=fs.readFileSync('docs/app.js','utf8');
const D=JSON.parse(fs.readFileSync('docs/data.json'));
const model=src.split('// ---------------------------------------------------------------- 3D view')[0].replace(/^import .*$/m,'');
const solver=src.slice(src.indexOf('// stage goals'),src.indexOf('const FACE_MOVES'));
const M=new Function('D',model+`const STAGE_TEXT={};let DATA=D;STICKERS=D.stickers;STICKERS.forEach(([p,n],i)=>INDEX.set(key(p,n),i));
FACE_OF=STICKERS.map(([,n])=>{const ax=n.findIndex(v=>v);return Object.keys(FACES).find(f=>FACES[f][0]===ax&&FACES[f][1]===n[ax]);});buildRotations();`+solver+`return {solveWith,applySeq,normalize};`)(D);
const FM=["R","L","U","D","F","B"].flatMap(f=>[f,f+"'",f+"2"]);
let seed=7; const rnd=()=>(seed=(seed*1103515245+12345)%2147483648)/2147483648;
const N=+process.argv[2]||300, cubes=[];
for(let k=0;k<N;k++){const mv=[];let last='';while(mv.length<25){const m=FM[Math.floor(rnd()*18)];if(m[0]===last)continue;mv.push(m);last=m[0];}cubes.push(M.applySeq(D.stickers.map((_,i)=>i),mv));}
for(const kind of['stages','full'])for(const a of D.arms){const out=[];for(const s of[0,1,2]){let ok=0;for(const c of cubes){const r=M.solveWith(kind,a.id,s,c);if(r.stuck)continue;
 let st=M.normalize(c);r.moves.forEach((m,i)=>{st=M.applySeq(st,[m]);if(r.bounds.includes(i+1))st=M.normalize(st);});st=M.normalize(st);
 if(!st.every((v,i)=>v===i)){console.log('BAD',kind,a.id,s,JSON.stringify(r.parts).slice(0,600));process.exit(1)};ok++;}out.push((100*ok/N).toFixed(0)+'%');}
 console.log(kind.padEnd(7),a.id.padEnd(11),out.join(' / '));}
