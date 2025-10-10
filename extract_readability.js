// extract_article.js
const { Readability } = require('@mozilla/readability');
const { JSDOM } = require('jsdom');

const fs = require('fs');

(async () => {
  const url = process.argv[2];
  const html = fs.readFileSync(process.argv[3], 'utf-8');
  
  const dom = new JSDOM(html, { url: url });
  const reader = new Readability(dom.window.document);
  const article = reader.parse();

  console.log(JSON.stringify(article, null, 2));
})();

