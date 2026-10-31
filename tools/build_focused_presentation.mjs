/* Editable defense deck from frozen evidence. No training or inference. */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';

const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const RUNTIME=path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies');
process.env.RUNTIME_NODE_MODULES=path.join(RUNTIME,'node/node_modules');
const NODE=path.join(RUNTIME,'node/bin/node.exe');
const SKILL=path.join(process.env.USERPROFILE,'.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11618/skills/presentations');
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(RUNTIME,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')));
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')));
const BUILD=path.join(ROOT,'tmp/presentation/focused');
const ASSET=path.join(ROOT,'docs/assets/presentation/focused');
const APP_CAPTURES=path.join(ROOT,'docs/assets/current-app-screenshots');
const COVER=JSON.parse(await fs.readFile(path.join(ROOT,'docs/academic-cover.json'),'utf8'));
await fs.mkdir(BUILD,{recursive:true}); await fs.mkdir(ASSET,{recursive:true});
const HASH={training:'e4f8b8f7c73693f713ecef69a06434ce75a3a101d2541654d7009398c4eb5624',study:'a4d5a849612f06eedef96e0d28e08247e5e3757eba0b4f2e4f1a893e9d4b8247',diagnostics:'b4c2c76e481e0f30e323ea22f9359212bac1b2018605f833682875d98face500',forecast:'fae5c875b1be148d7a069259b19df9b04666fe97e2182ce8898f45fed7b0a43e'};
async function readPinned(name,hash){const b=await fs.readFile(path.join(ROOT,'data/evidence',name));if(crypto.createHash('sha256').update(b).digest('hex')!==hash)throw Error(`Hash mismatch: ${name}`);return JSON.parse(b);}
const training=await readPinned('focused-ppo-training.json',HASH.training);
const study=await readPinned('focused-ppo-study.json',HASH.study);
const diagnostics=await readPinned('focused-ppo-training-diagnostics.json',HASH.diagnostics);
const forecast=(await readPinned('proposal-supplement.json',HASH.forecast)).forecast.methods;
if(diagnostics.training_sha256!==HASH.training || study.training_sha256!==HASH.training)throw Error('Training identity mismatch');
const SEEDS=training.runs.map(r=>r.seed);
const P=study.procedure_means.PPO,F=study.controller_means['Fixed Rotation'],O=study.controller_means['Constrained Optimization'];
const C={bg:'#F5F8F5',ink:'#173F43',teal:'#137B73',coral:'#E77D60',muted:'#58787A',line:'#D8E5DF',soft:'#E7F0EC',white:'#FFFFFF'};
const seedColors=['#7DB8B2','#E9B7A9','#A5B4D1','#BEB0C9','#CCBB8F'];
const FONT='Atkinson Hyperlegible';
const pres=Presentation.create({slideSize:{width:1280,height:720}});
const nativeCharts=[],nativeTables=[];
let serial=0;
function box(s,x,y,w,h,fill=C.white,line='none'){return s.shapes.add({geometry:'rect',name:`shape-${++serial}`,position:{left:x,top:y,width:w,height:h},fill,line:{fill:line,width:line==='none'?0:1}});}
function text(s,str,x,y,w,h,size=26,color=C.ink,bold=false,align='left'){
 const q=box(s,x,y,w,h,'none');q.text=str;q.text.style={typeface:FONT,fontSize:size,bold,color,alignment:align,verticalAlignment:'top',insets:0,wrap:'square',autoFit:'none'};return q;
}
function slide(title,sub='',citation=''){
 const s=pres.slides.add();s.background.fill=C.bg;const n=pres.slides.items.length;
 text(s,'GridGuard Khartoum',64,28,650,24,18,C.teal,true);
 text(s,title,64,77,1152,64,42,C.ink,true);
 if(sub)text(s,sub,64,143,1152,51,23,C.muted);
 box(s,64,677,1152,1,C.line);text(s,citation||'Synthetic scenarios. Decision support only.',64,687,1080,20,14,C.muted);
 text(s,String(n).padStart(2,'0'),1155,684,61,23,17,C.teal,false,'right');
 s.speakerNotes.text=`GridGuard Khartoum study. Training SHA-256 ${HASH.training}; study ${HASH.study}; diagnostics ${HASH.diagnostics}. ${citation}`;
 return s;
}
function note(s,str,y=610,color=C.teal){text(s,str,64,y,1152,54,23,color,true);}
function lines(s,items,x,y,w,gap=76,size=26){items.forEach((t,i)=>{box(s,x,y+i*gap+12,6,27,C.coral);text(s,t,x+23,y+i*gap,w-23,gap-8,size);});}
function table(s,headers,rows,{x=64,y=220,w=1152,h=330,widths}={}){
 const data=[headers,...rows];const t=s.tables.add({rows:data.length,columns:headers.length,left:x,top:y,width:w,height:h,values:data,columnWidths:widths});
 const all=t.cells.block({row:0,column:0,rowCount:data.length,columnCount:headers.length});
 all.assign({textStyle:{typeface:FONT,fontSize:21,color:C.ink},margins:{left:12,right:10,top:11,bottom:9},borders:{fill:C.line,width:0.6}});
 for(let r=0;r<data.length;r++)for(let c=0;c<headers.length;c++){let cell=t.getCell(r,c);cell.fill=r===0?C.teal:r%2?C.white:C.soft;cell.text.style={typeface:FONT,fontSize:21,color:r===0?C.white:C.ink,bold:r===0,insets:10};}
 nativeTables.push(pres.slides.items.length);return t;
}
function chart(s,{x,y,w,h,title,categories,series,type='line',min,max,format='0.0',legend=false,yTitle='',xTitle='',majorUnit}){
 // Excel-backed presentation snapshots retain 15 significant digits; frozen inputs stay unchanged.
 series=series.map(item=>({...item,values:item.values.map(value=>Number(value.toPrecision(15)))}));
 const cfg={position:{left:x,top:y,width:w,height:h},title,titleTextStyle:{fontSize:22,bold:true,fill:C.ink},categories,series,hasLegend:legend,legend:{position:'bottom',overlay:false,textStyle:{fontSize:16,fill:C.muted}},chartFill:C.bg,chartLine:{fill:'none',width:0},plotAreaFill:C.bg,plotAreaLine:{fill:'none',width:0},xAxis:{textStyle:{fontSize:16,fill:C.muted},line:{fill:C.line,width:1},title:xTitle?{text:xTitle,textStyle:{fontSize:16,fill:C.muted}}:undefined},yAxis:{min,max,majorUnit,numberFormatCode:format,textStyle:{fontSize:16,fill:C.muted},majorGridlines:{fill:C.line,width:0.7},title:yTitle?{text:yTitle,textStyle:{fontSize:17,fill:C.ink}}:undefined},lineOptions:{smooth:false},dataLabels:{showValue:false}};
 if(type==='bar'){cfg.barOptions={direction:'column',grouping:'clustered',gapWidth:100};cfg.dataLabels={showValue:true,position:'outEnd',numberFormatCode:format,textStyle:{fontSize:17,fill:C.ink}};}
 if(type==='line'){
  cfg.series=series.map(item=>({...item,xValues:categories.map(Number),smooth:false}));
  cfg.scatterOptions={style:'line',varyColors:false};
  cfg.xAxis={...cfg.xAxis,min:0,max:3000,majorUnit:1000,numberFormatCode:'0',tickLabelPosition:'low'};
  type='scatter';
 }
 const ch=s.charts.add(type,cfg);applyPresentationChartFont(ch,{fontFamily:FONT});nativeCharts.push(pres.slides.items.length);return ch;
}
async function picture(s,p,x,y,w,h,alt){const b=await fs.readFile(p);return s.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:/\.jpe?g$/i.test(p)?'image/jpeg':'image/png',alt,fit:'contain',position:{left:x,top:y,width:w,height:h}});}
const equations={
 constraint:String.raw`C+\sum_{i=1}^{7}x_i b_i\leq S,\qquad x_i\in\{0,1\}`,
 dispatch:String.raw`\alpha=\min\!\left(1,\frac{S}{\max(Q,\varepsilon)}\right),\qquad \mathrm{delivered}=\alpha\,\mathrm{requested}`,
 reward:String.raw`r_t=\frac{5C_t+2U_t+2J_t-0.25W_t-3O_t}{9}`,
 jain:String.raw`J_t=\frac{\left(\sum_{i=1}^{7}z_{i,t}\right)^2}{7\sum_{i=1}^{7}z_{i,t}^{2}+\varepsilon}`,
 ratio:String.raw`\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}`,
 loss:String.raw`\mathcal L=-\mathbb E_t\!\left[\min\!\left(\rho_t\widehat A_t,\operatorname{clip}(\rho_t,0.8,1.2)\widehat A_t\right)\right]+0.25\mathbb E_t[(V_\phi-\widehat R_t)^2]-c_H\mathbb E_t[H(\pi_\theta)]`,
 horizons:String.raw`a_t^{\mathrm{opt}}=\operatorname{arg\,max}_{a\in\mathcal A(s_t)}r_t(s_t,a),\qquad \pi^{\mathrm{PPO}}:\ \max\mathbb E\!\left[\sum_{t=0}^{23}r_t\right]`,
 variance:String.raw`\bar x_j=\frac{1}{5}\sum_{k=1}^{5}x_{k,j},\qquad s_j^2=\frac{1}{4}\sum_{k=1}^{5}(x_{k,j}-\bar x_j)^2`,
 ar:String.raw`\widehat d_t=\max\!\left(0,\beta_0+\beta_1d_{t-1}+\beta_2d_{t-2}+\beta_3d_{t-3}\right)`
};
const mathRequest={directory:path.join(ASSET,'math'),manifest:path.join(ASSET,'math-manifest.json'),expressions:Object.entries(equations).map(([id,latex])=>({id,latex,display:true}))};
await fs.writeFile(path.join(BUILD,'math-request.json'),JSON.stringify(mathRequest));
execFileSync(NODE,[path.join(ROOT,'tools/render_thesis_math.cjs'),path.join(BUILD,'math-request.json')],{stdio:'inherit'});
const math=JSON.parse(await fs.readFile(mathRequest.manifest));
async function eq(s,id,x,y,w,h){await picture(s,math[id].png,x,y,w,h,`LaTeX: ${equations[id]}`);}
const fixedCI=study.primary_family_adjusted_inference['PPO - Fixed Rotation'];
const optCI=study.primary_family_adjusted_inference['PPO - Constrained Optimization'];
const signed=(v,d=3)=>(v>=0?'+':'')+v.toFixed(d);
const ci=(v,scale=1,d=3)=>`${signed(v.mean*scale,d)}  [${signed(v.lower*scale,d)}, ${signed(v.upper*scale,d)}]`;

// 1
{
 const s=pres.slides.add();s.background.fill=C.white;
 await picture(s,path.join(ROOT,COVER.logos),240,0,800,172,'Faculty of Mathematical Sciences and Informatics and University of Khartoum logos');
 text(s,COVER.university,80,165,1120,34,26,C.ink,true,'center');
 text(s,COVER.faculty,80,205,1120,30,22,C.ink,false,'center');
 text(s,COVER.project_name,80,252,1120,30,23,C.teal,true,'center');
 text(s,COVER.project_title,80,292,1120,94,36,C.ink,true,'center');
 text(s,COVER.cover_statement,80,402,1120,70,22,C.ink,false,'center');
 text(s,COVER.authors.map(a=>`${a.name} (${a.index})`).join('\n'),80,492,1120,93,23,C.ink,false,'center');
 text(s,`Project Supervisor: ${COVER.supervisor}`,80,610,1120,32,23,C.ink,false,'center');
 text(s,COVER.submission_date,80,661,1120,30,22,C.ink,true,'center');
 s.speakerNotes.text=`${COVER.project_title}. ${COVER.cover_statement} Project supervisor: ${COVER.supervisor}. ${COVER.submission_date}. Original university and faculty logo image supplied in the academic reference. Cover metadata: docs/academic-cover.json.`;
}
// 2
{
 const s=slide('Hourly shortage allocation in Khartoum','Seven localities, five sectors and a shared supply pool','Context: World Bank (2025). Related control research: Chen et al. (2023).');
 lines(s,['Protect Hospital, Water and Bakery demand.','Allocate remaining supply to Residential and Commercial demand.','Measure service, interruption duration and distribution efficiency.'],64,223,580,112,29);
 await picture(s,path.join(APP_CAPTURES,'overview.png'),670,210,545,370,'Current desktop overview and locality map');
 note(s,'All demand and supply traces are synthetic. The map shows administrative geography.',612);
 s.speakerNotes.text+=' https://documents1.worldbank.org/curated/en/099050125112514628/pdf/P507788-6a843250-44f5-47da-9f07-c6b1269a091f.pdf; https://doi.org/10.1016/j.egyr.2023.03.027. Screenshot: current desktop overview. Live Scenario uses the constrained PPO actor from seed 8111, selected checkpoint 2500. Live scenario values remain separate from the frozen five-run study averages.';
}
// 3
{
 const s=slide('The simulation and service choices','24 hourly decisions with current observations and accumulated service history');
 table(s,['Service tier','Requested demand'],[['Off','No sectors'],['Critical Only','Hospital, Water and Bakery'],['Full Service','All five sectors']],{x:64,y:218,w:560,h:236,widths:[190,370]});
 table(s,['Shortage','Supply construction'],[['Mild','80% of total demand'],['Moderate','65%, minus 10 pp at hours 17–20'],['Severe','50%, minus 15 pp at hours 16–21']],{x:655,y:218,w:561,h:236,widths:[150,411]});
 lines(s,['Independent demand and supply perturbations within ±5%.','One supply pool. No network power flow, storage or restoration transients.'],64,490,1150,66,25);
}
// 4
{
 const s=slide('Critical protection through feasible actions','PPO and Constrained Optimization share 128 Critical Only / Full Service combinations','Masking: Huang and Ontañón (2022). Safety distinction: García and Fernández (2015).');
 text(s,'Critical reservation',64,218,570,42,28,C.teal,true);await eq(s,'constraint',66,275,1120,66);
 text(s,'C: total critical demand     bᵢ: locality noncritical demand     S: available supply',64,357,1152,42,24,C.muted);
 await eq(s,'dispatch',66,421,1120,73);
 text(s,'Shared proportional dispatch uses ε = 10⁻⁸. If critical demand cannot fit, the all-Critical-Only fallback still records the resulting interruption.',64,524,1152,85,26);
}
// 5
{
 const s=slide('Three decision procedures','The optimizer and PPO receive equivalent causal information and the same feasible set');
 table(s,['Controller','Decision rule','Horizon'],[['Fixed Rotation','Three localities Full Service for each three-hour block','Fixed schedule'],['Constrained Optimization','Checks 128 assignments and maximizes current reward over the feasible set','Current hour'],['PPO','Chooses the highest-logit feasible action from a learned policy','Daily training objective']],{y:225,h:286,widths:[275,610,267]});
 await eq(s,'horizons',68,549,1140,65);
}
// 6
{
 const s=slide('Reward and public service measures','The training objective combines protection, utilization, fairness and switching','Fairness definition: Jain, Chiu and Hawe (1984).');
 await eq(s,'reward',120,211,1040,76);
 text(s,'C: hourly critical service     U: supply utilization     J: cumulative-service fairness\nW: fraction changing tier     O: excess requested power / total demand',64,311,1152,76,25);
 await eq(s,'jain',360,405,560,88);
 text(s,'zᵢ,t is cumulative served energy / cumulative demand. Report mean hourly Jain fairness, direct duration measures and energy separately from daily reward.',64,527,1152,85,27);
}
// 7
{
 const s=slide('Masked PPO training','Separate actor and critic networks: 46 inputs, two 256-unit tanh layers, 128 actor logits','PPO: Schulman et al. (2017). Masking: Huang and Ontañón (2022).');
 await eq(s,'ratio',380,215,520,85);await eq(s,'loss',64,332,1152,98);
 text(s,'The same stored feasibility mask defines both policy probabilities.\nAdvantages normalize over the full rollout. Value targets remain fixed.',64,463,1152,80,27);
 text(s,'γ = 1.0, GAE λ = 0.95, zero terminal value. The critic term is exactly 0.25 MSE. Entropy coefficient decreases from 0.01 to 0.001.',64,568,1152,68,24,C.muted);
}
// 8
{
 const s=slide('Five independent runs and frozen selection','Fresh initialization, disjoint training streams and a fixed training budget');
 table(s,['Training element','Configuration'],[['Training seeds','1729, 2718, 3141, 5772, 8111'],['Each run','3,000 iterations × 256 episodes × 24 hours'],['Total PPO exposure','92,160,000 transitions'],['Initialization','Fairness Rule demonstrations: 32,760 states, 10 epochs'],['PPO updates','4 epochs, minibatch 1,024, learning rate 0.0003 to zero']],{y:213,h:310,widths:[330,822]});
 text(s,'Select the best validation reward among checkpoints with zero critical/power violations and fairness/utilization at least Fixed Rotation overall and at every shortage level.',64,554,1152,83,26);
}
// 9
{
 const s=slide('Matched evaluation and uncertainty','All final-test decisions follow the frozen five-run validation selection','Statistical reliability: Agarwal et al. (2021).');
 table(s,['Split','Independent base draws','Coupled conditions'],[['Validation','5 seeds × 32 = 160','480'],['Final test','10 seeds × 32 = 320','960']],{y:225,h:180,widths:[245,520,387]});
 lines(s,['12 methods: Fixed Rotation, optimizer, five selected PPO actors and five iteration-600 actors.','11,520 trajectories. Bootstrap resamples five training runs and matched base-draw clusters.','10,000 resamples. Two primary claims use adjusted 97.5% intervals.'],64,437,1152,68,25);
}
// 10
{
 const s=slide('Main outcomes on 960 matched conditions','PPO values average the five independently trained policies equally');
 for(const [i,m,title,scale,max] of [[0,'critical_interruption_locality_hours','Critical interruption',1,120],[1,'fairness','Jain service fairness (%)',100,110],[2,'utilization','Supply utilization (%)',100,110]]){
 chart(s,{x:64+i*390,y:219,w:372,h:340,title,categories:['Fixed','Optimizer','PPO'],series:[{name:title,values:[F[m],O[m],P[m]].map(v=>v*scale),fill:C.teal,points:[{idx:0,fill:'#718789'},{idx:1,fill:C.coral},{idx:2,fill:C.teal}]}],type:'bar',min:0,max,format:'0.00',yTitle:i===0?'Locality-hours / day':''});}
 note(s,'PPO and the optimizer preserve critical service. Their remaining differences concern allocation.',596);
}
// 11
{
 const s=slide('PPO improves service over Fixed Rotation','Paired effects are PPO minus Fixed Rotation. Intervals have adjusted 97.5% coverage.');
 table(s,['Outcome','Effect [lower, upper]'],[['Critical interruption (locality-hours)',ci(fixedCI.critical_interruption_locality_hours,1,3)],['Jain fairness (percentage points)',ci(fixedCI.fairness,100,3)],['Utilization (percentage points)',ci(fixedCI.utilization,100,3)]],{y:235,h:260,widths:[610,542]});
 text(s,'Full declared comparison: PASSED',64,536,1152,47,34,C.teal,true);
 text(s,'This compares the complete constrained system with a supply-blind fixed schedule. The critical-service rule contributes to the improvement.',64,596,1152,61,25);
}
// 12
{
 const s=slide('A reward and fairness gain with a utilization tradeoff','Paired effects are PPO minus Constrained Optimization. Adjusted 97.5% intervals.');
 table(s,['Outcome','Effect [lower, upper]'],[['Daily reward',ci(optCI.episode_reward,1,4)],['Jain fairness (percentage points)',ci(optCI.fairness,100,3)],['Utilization (percentage points)',ci(optCI.utilization,100,3)],['Switching (percentage points)',ci(optCI.switching_rate,100,3)]],{y:221,h:306,widths:[610,542]});
 text(s,'Full service-guarded improvement criterion: NOT MET',64,558,1152,44,31,C.coral,true);
 text(s,'The utilization loss exceeds the allowed 0.2-point margin. Both protect critical service.',64,610,1152,43,25);
}
// 13
{
 const s=slide('Duration fairness and total interruption burden','Episode duration measures are calculated before averaging');
 const rows=[['Zero-service blackout','blackout_locality_hours','locality-hours'],['Blackout pairwise gap','blackout_duration_mean_pairwise_gap_hours','hours'],['Critical-interruption pairwise gap','critical_interruption_duration_mean_pairwise_gap_hours','hours'],['Longest critical interruption','longest_critical_interruption_hours','hours']].map(([label,key,unit])=>[`${label} (${unit})`,F[key].toFixed(2),O[key].toFixed(2),P[key].toFixed(2)]);
 table(s,['Measure','Fixed Rotation','Optimizer','PPO'],rows,{y:224,h:305,widths:[560,212,190,190]});
 text(s,'Critical Only removes locality-wide zero-service blackouts while Residential and Commercial demand can remain unmet. Zero duration does not mean Full Service.',64,570,1152,86,27);
}
// 14
{
 const s=slide('Shortage-specific allocation outcomes','Results keep the Mild, Moderate and Severe conditions visible');
 for(const [i,key,label]of [[0,'fairness','Jain service fairness (%)'],[1,'utilization','Supply utilization (%)']]){
 const series=[['Fixed Rotation',C.muted],['Constrained Optimization',C.coral],['PPO',C.teal]].map(([name,color])=>({name:name==='Constrained Optimization'?'Optimizer':name,values:['mild','moderate','severe'].map(level=>(name==='PPO'?study.procedure_shortage_means[level].PPO[key]:study.shortage_means[level][name][key])*100),fill:color}));
 chart(s,{x:64+i*580,y:221,w:560,h:360,title:label,categories:['Mild','Moderate','Severe'],series,type:'bar',min:0,max:110,format:'0.0',legend:true});}
 note(s,'Critical interruption is zero for PPO and the optimizer at every tested shortage level.',615);
}
// 15
{
 const s=slide('Validation progress across all five training runs','31 checkpoints per run. Checkpoint selection never consults final-test performance.');
 for(const [i,key,label,scale,min,max]of [[0,'episode_reward','Daily reward',1,23.2,23.55],[1,'fairness','Jain fairness (%)',100,92,98],[2,'utilization','Utilization (%)',100,96.5,99]]){
 const series=training.runs.map((r,j)=>({name:String(r.seed),values:r.checkpoints.map(p=>p.summary.overall[key]*scale),line:{fill:seedColors[j],width:2},marker:{symbol:'none'}}));
 chart(s,{x:64+i*390,y:226,w:372,h:356,title:label,categories:training.runs[0].checkpoints.map(p=>String(p.iteration)),series,min,max,format:i?'0.0':'0.00',xTitle:'PPO iteration'});}
 text(s,'Seeds: 1729   2718   3141   5772   8111',64,605,1152,37,22,C.muted);
}
// 16 and 17
for(const [keys,title]of [[['episode_reward','policy_loss','value_loss'],'Training reward and loss'],[['entropy','approx_kl','clip_fraction'],'Policy entropy and update diagnostics']]){
 const s=slide(title,'120 recorded updates per seed, every 25 iterations. No smoothing.');
 const names={episode_reward:'Rollout daily reward',policy_loss:'Policy loss',value_loss:'Value loss (half MSE)',entropy:'Policy entropy (nats)',approx_kl:'Approximate KL',clip_fraction:'Clipped fraction'};
 keys.forEach((key,i)=>{
 const series=training.runs.map((r,j)=>({name:String(r.seed),values:r.telemetry.slice(1).map(p=>p[key]),line:{fill:seedColors[j],width:0.8},marker:{symbol:'none'}}));
 const points=diagnostics.diagnostics.map(p=>p.metrics[key]);
 series.push({name:'Five-run mean',values:points.map(p=>p.mean),line:{fill:C.teal,width:2.8},marker:{symbol:'none'}});
 for(const sign of [-1,1])series.push({name:sign<0?'Mean - sample SD':'Mean + sample SD',values:points.map(p=>p.mean+sign*p.sample_sd),line:{fill:'#69A49D',style:'dashed',width:1.3},marker:{symbol:'none'}});
 chart(s,{x:64+i*390,y:223,w:372,h:338,title:names[key],categories:diagnostics.diagnostics.map(p=>String(p.iteration)),series,format:['value_loss','approx_kl','policy_loss'].includes(key)?'0.0000':'0.00',xTitle:'PPO iteration'});
 });
 text(s,'Thin lines: five seeds. Dark line: mean. Dashed limits: mean ± sample SD, not a CI.',64,581,1152,44,24,C.teal,true);
 text(s,keys[0]==='episode_reward'?'Reward describes the pre-update rollout. Losses average all minibatches of that update’s four epochs.':'Entropy, KL and clipping average all minibatches of the logged update. They are not 25-update moving averages.',64,625,1152,40,20,C.muted);
}
// 18
{
 const s=slide('Between-seed validation variance','Sample variance of the five run means on the same validation scenarios');
 for(const [i,key,label]of [[0,'episode_reward','Reward variance (score²)'],[1,'fairness','Fairness variance (pp²)'],[2,'utilization','Utilization variance (pp²)']]){
 chart(s,{x:64+i*390,y:220,w:372,h:310,title:label,categories:diagnostics.validation.map(p=>String(p.iteration)),series:[{name:'Sample variance',values:diagnostics.validation.map(p=>p.metrics[key].sample_variance),line:{fill:C.teal,width:2.4},marker:{symbol:'none'}}],min:0,format:key==='episode_reward'?'0.0000':'0.000',xTitle:'PPO iteration'});}
 await eq(s,'variance',310,542,660,71);
 text(s,'31 checkpoints include behavior-cloning iteration 0. This is between-run variance, not critic explained variance or final-test uncertainty.',64,621,1152,45,22,C.muted);
}
// 19
{
 const s=slide('Longer training improves reward but loses utilization','Same-run iteration-600 snapshots are a within-schedule reference');
 const rows=training.runs.map(r=>{const a=study.controller_means[`PPO seed ${r.seed}`].episode_reward,b=study.controller_means[`PPO600 seed ${r.seed}`].episode_reward;return[String(r.seed),String(r.selected.iteration),b.toFixed(4),a.toFixed(4),signed(a-b,4)];});
 table(s,['Seed','Selected iteration','Reward at 600','Selected reward','Difference'],rows,{y:220,h:300,widths:[150,240,250,250,262]});
 text(s,'Mean daily reward +0.0295. Utilization −0.2417 percentage points.',64,554,1152,48,28,C.teal,true);
 text(s,'The full additional-training criterion fails its utilization and shortage guardrails. The 600 snapshots use the planned 3,000-iteration schedule.',64,609,1152,52,23,C.coral);
}
// 20
{
 const s=slide('Learned next-hour demand prediction','Ridge AR(3): 768 training traces and 320 separate evaluation traces');
 await eq(s,'ar',220,207,840,64);
 const methods=['trailing_mean_3h','persistence_1h','ridge_ar3'];
 for(const [i,scope,key,title]of [[0,'locality_sector','mae_mw','Cell MAE (MW)'],[1,'system_total','mae_mw','System MAE (MW)'],[2,'locality_sector','rmse_mw','Cell RMSE (MW)'],[3,'system_total','rmse_mw','System RMSE (MW)']]){
 chart(s,{x:64+i*290,y:310,w:274,h:270,title,categories:['3h mean','Persist.','Ridge'],series:[{name:title,values:methods.map(m=>forecast[m][scope][key]),points:[{idx:0,fill:'#718789'},{idx:1,fill:C.coral},{idx:2,fill:C.teal}]}],type:'bar',min:0,max:i%2?60:5,format:i%2?'0.0':'0.00'});}
 note(s,'Forecasts use past demand and identical target hours. Prediction does not enter PPO.',617);
}
// 21
{
 const s=slide('Current study results in the desktop app','','Synthetic decision support. Live Scenario uses one selected actor. Results show the five-run study.');
 await picture(s,path.join(APP_CAPTURES,'results.png'),160,137,960,540,'Current desktop Results screen with the frozen five-run study');
 s.speakerNotes.text+=' Screenshot: current desktop Results. Live Scenario uses the constrained PPO actor from seed 8111, selected checkpoint 2500, and the study constrained optimizer. The Results page retains the five-run aggregate on 960 matched conditions. This single live policy and an individual replay do not replace those averages. The complete scientific comparison and its utilization limitation remain on slides 12 and 22.';
}
// 22
{
 const s=slide('Conclusions and the next engineering step');
 text(s,'Against Fixed Rotation',64,198,550,40,30,C.teal,true);text(s,'The constrained PPO system improves critical service, fairness and utilization on fresh matched synthetic scenarios.',64,253,550,125,30);
 text(s,'Against the optimizer',665,198,550,40,30,C.coral,true);text(s,'PPO improves fairness and daily reward but uses less supply and switches more often. The full criterion fails.',665,253,550,125,30);
 box(s,64,411,1152,1,C.line);
 lines(s,['Ridge AR(3) improves prediction over the two simple baselines.','Earlier Fairness Rule comparisons remain unfavorable for PPO.','Measured demand and feasible feeder actions are needed before operational evaluation.'],64,449,1152,64,25);
}
// 23 and 24: alphabetized Harvard bibliography, only works actually cited.
const references=[
 'Agarwal, R., Schwarzer, M., Castro, P.S., Courville, A. and Bellemare, M.G. (2021) “Deep reinforcement learning at the edge of the statistical precipice”, NeurIPS 34. https://arxiv.org/abs/2108.13264',
 'Chen, H., Zhuang, J., Zhou, G., Wang, Y., Sun, Z. and Levron, Y. (2023) “Emergency load shedding strategy for high renewable energy penetrated power systems based on deep reinforcement learning”, Energy Reports, 9, pp. 434–443. https://doi.org/10.1016/j.egyr.2023.03.027',
 'García, J. and Fernández, F. (2015) “A comprehensive survey on safe reinforcement learning”, JMLR, 16(42), pp. 1437–1480. https://jmlr.org/papers/v16/garcia15a.html',
 'Huang, S. and Ontañón, S. (2022) “A closer look at invalid action masking in policy gradient algorithms”, FLAIRS Conference Proceedings, 35. https://doi.org/10.32473/flairs.v35i.130584',
 'Jain, R., Chiu, D.-M. and Hawe, W.R. (1984) A quantitative measure of fairness and discrimination for resource allocation in shared computer systems. DEC Research Report TR-301.',
 'Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017) “Proximal policy optimization algorithms”. https://doi.org/10.48550/arXiv.1707.06347',
 'World Bank (2025) Accelerating Sustainable and Clean Energy and Digital Access Transformation in Sudan (P507788). Project Information Document, appraisal stage, PIDIA01286, 1 May. documents.worldbank.org, project P507788.'
];
const referenceItalics=['NeurIPS 34','Energy Reports','JMLR','FLAIRS Conference Proceedings','A quantitative measure of fairness and discrimination for resource allocation in shared computer systems','Proximal policy optimization algorithms','Accelerating Sustainable and Clean Energy and Digital Access Transformation in Sudan'];
function referenceRuns(r){const part=referenceItalics.find(q=>r.includes(q));const start=r.indexOf(part);return [[{run:r.slice(0,start)},{run:part,style:{italic:true}},{run:r.slice(start+part.length)}]];}
for(const [i,refs]of [references.slice(0,4),references.slice(4)].entries()){
 const s=slide(`References ${i+1} of 2`,'Alphabetized author-date references');
 refs.forEach((r,j)=>text(s,referenceRuns(r),64,215+j*108,1152,98,23));
 s.speakerNotes.text=refs.join('\n\n');
}
if(pres.slides.items.length!==24)throw Error('Unexpected slide count');
await fs.writeFile(path.join(BUILD,'source-binding.json'),JSON.stringify({hashes:HASH,slide_count:24,native_charts:nativeCharts,native_tables:nativeTables,math_manifest:mathRequest.manifest},null,2));
const candidate=path.join(BUILD,'candidate.pptx');
await (await PresentationFile.exportPptx(pres)).save(candidate);
console.log('Exported candidate',candidate);
// Artifact previews are optional on hosts whose native canvas renderer is unavailable.
if(process.argv.includes('--preview'))for(const [i,s]of pres.slides.items.entries()){
 const b=await pres.export({slide:s,format:'png',scale:1});await fs.writeFile(path.join(BUILD,`artifact-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));
}
const finalStamp=Date.now();
const final=path.join(BUILD,`validated-${finalStamp}.pptx`);
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:final,pythonExecutable:path.join(RUNTIME,'python/python.exe'),integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...[...new Set(nativeTables)].flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:24,requiredNativeChartOwnerSlides:[...new Set(nativeCharts)],requiredNativeTableOwnerSlides:[...new Set(nativeTables)],materializeLiteralChartWorkbooks:true,nativeChartTargetApplication:'portable',fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:path.join(ROOT,`tmp/presentation/focused-validation-${finalStamp}.json`)});
await fs.writeFile(path.join(BUILD,'validated-path.txt'),final);
console.log('Validated PPTX',final,JSON.stringify(result));
