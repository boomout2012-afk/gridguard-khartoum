/* Render the thesis's LaTeX expressions with MathJax, independent of the GPU study. */
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const root = path.resolve(__dirname, '..');
const requireMath = createRequire(path.join(root, 'tmp/math-typesetting/package.json'));
const {mathjax} = requireMath('mathjax-full/js/mathjax.js');
const {TeX} = requireMath('mathjax-full/js/input/tex.js');
const {SVG} = requireMath('mathjax-full/js/output/svg.js');
const {liteAdaptor} = requireMath('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = requireMath('mathjax-full/js/handlers/html.js');
const {AllPackages} = requireMath('mathjax-full/js/input/tex/AllPackages.js');
const sharp = require(process.env.GRIDGUARD_SHARP_PATH ||
  path.join(process.env.USERPROFILE, '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp'));
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({packages: AllPackages, formatError(_jax, error) { throw error; }});
const document = mathjax.document('', {InputJax: tex, OutputJax: new SVG({fontCache: 'none'})});

async function main() {
  const request = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  fs.mkdirSync(request.directory, {recursive: true});
  const rendered = {};
  for (const item of request.expressions) {
    const node = document.convert(item.latex, {display: item.display});
    const svg = adaptor.tags(node, 'svg')[0];
    const widthEx = parseFloat(adaptor.getAttribute(svg, 'width'));
    const heightEx = parseFloat(adaptor.getAttribute(svg, 'height'));
    const style = adaptor.getAttribute(svg, 'style') || '';
    const alignEx = Number((style.match(/vertical-align:\s*([-\d.]+)ex/) || [0, 0])[1]);
    const fontSize = item.display ? 11.5 : 10.5;
    const width = widthEx * 0.442 * fontSize;
    const height = heightEx * 0.442 * fontSize;
    const baseline = alignEx * 0.442 * fontSize;
    adaptor.setAttribute(svg, 'width', `${Math.ceil(width * 600 / 72)}px`);
    adaptor.setAttribute(svg, 'height', `${Math.ceil(height * 600 / 72)}px`);
    const markup = adaptor.outerHTML(svg).replaceAll('currentColor', '#173F43');
    const stem = path.join(request.directory, item.id);
    fs.writeFileSync(stem + '.svg', markup);
    await sharp(Buffer.from(markup)).png().toFile(stem + '.png');
    rendered[item.id] = {latex: item.latex, display: item.display, width, height, baseline,
      png: stem + '.png', svg: stem + '.svg'};
  }
  fs.writeFileSync(request.manifest, JSON.stringify(rendered, null, 2));
  console.log(`Rendered ${Object.keys(rendered).length} LaTeX expressions.`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
