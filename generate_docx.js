const fs = require('fs');
const path = require('path');
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  HeadingLevel,
  AlignmentType,
  BorderStyle,
  WidthType,
  ShadingType,
} = require('docx');

const mdPath = path.join(__dirname, 'research_paper.md');
const mdContent = fs.readFileSync(mdPath, 'utf-8');

const docChildren = [];

// Title
docChildren.push(
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 240, after: 120 },
    children: [
      new TextRun({
        text: 'Zero-Latency Edge RAG: Decentralized Client-Side Vector Search via WebGPU Compute Shaders and OffscreenCanvas Spatial Thread Isolation',
        bold: true,
        size: 36, // 18pt
        font: 'Times New Roman',
      }),
    ],
  })
);

// Authors
docChildren.push(
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [
      new TextRun({
        text: 'Principal Systems Research Scientist',
        bold: true,
        size: 24, // 12pt
        font: 'Times New Roman',
      }),
      new TextRun({
        text: '\nAdvanced Agentic Coding & High-Performance Browser Architectures',
        italic: true,
        size: 22, // 11pt
        font: 'Times New Roman',
      }),
    ],
  })
);

// Divider
docChildren.push(
  new Paragraph({
    spacing: { after: 200 },
    border: { bottom: { color: '000000', space: 1, value: BorderStyle.SINGLE, size: 6 } },
  })
);

// Split Markdown into sections
const lines = mdContent.split('\n');
let inAbstract = false;
let abstractText = '';
let inCode = false;
let codeText = '';
let codeLang = '';
let inTable = false;
let tableRows = [];

for (let i = 0; i < lines.length; i++) {
  const line = lines[i];

  // Skip title header from MD
  if (line.startsWith('# Zero-Latency Edge RAG')) continue;
  if (line.startsWith('**Principal Systems Research Scientist**')) continue;
  if (line.startsWith('*Advanced Agentic Coding')) continue;
  if (line.trim() === '---') continue;

  // Code Block Handler
  if (line.startsWith('```')) {
    if (!inCode) {
      inCode = true;
      codeLang = line.replace('```', '').trim();
      codeText = '';
    } else {
      inCode = false;
      // Emit Code Block
      docChildren.push(
        new Paragraph({
          spacing: { before: 120, after: 120 },
          shading: { type: ShadingType.CLEAR, fill: 'F4F4F6' },
          border: {
            left: { color: '00F0FF', space: 4, value: BorderStyle.SINGLE, size: 12 },
          },
          children: [
            new TextRun({
              text: codeText.trimEnd(),
              font: 'Consolas',
              size: 19, // 9.5pt
              color: '1E1E1E',
            }),
          ],
        })
      );
    }
    continue;
  }

  if (inCode) {
    codeText += line + '\n';
    continue;
  }

  // Table Handler
  if (line.trim().startsWith('|')) {
    if (!line.includes('---')) {
      const cells = line.split('|').filter((_, idx, arr) => idx > 0 && idx < arr.length - 1).map(c => c.trim());
      tableRows.push(cells);
    }
    inTable = true;
    continue;
  } else if (inTable) {
    inTable = false;
    // Build docx Table
    if (tableRows.length > 0) {
      const docRows = tableRows.map((rowCells, rIdx) => {
        return new TableRow({
          children: rowCells.map(cellText => {
            // Clean markdown bolding
            const cleanText = cellText.replace(/\*\*/g, '').replace(/\$/g, '');
            return new TableCell({
              shading: rIdx === 0 ? { type: ShadingType.CLEAR, fill: '1B263B' } : undefined,
              margins: { top: 100, bottom: 100, left: 150, right: 150 },
              children: [
                new Paragraph({
                  alignment: rIdx === 0 ? AlignmentType.CENTER : AlignmentType.LEFT,
                  children: [
                    new TextRun({
                      text: cleanText,
                      bold: rIdx === 0,
                      color: rIdx === 0 ? 'FFFFFF' : '000000',
                      size: 19,
                      font: 'Times New Roman',
                    }),
                  ],
                }),
              ],
            });
          }),
        });
      });

      docChildren.push(
        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          rows: docRows,
        })
      );
      docChildren.push(new Paragraph({ spacing: { after: 120 } }));
      tableRows = [];
    }
  }

  // Abstract Block
  if (line.startsWith('### Abstract')) {
    inAbstract = true;
    docChildren.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_3,
        spacing: { before: 180, after: 60 },
        children: [
          new TextRun({
            text: 'Abstract',
            bold: true,
            size: 24,
            font: 'Times New Roman',
          }),
        ],
      })
    );
    continue;
  }

  if (inAbstract) {
    if (line.startsWith('**Keywords**')) {
      inAbstract = false;
      const kw = line.replace('**Keywords**—', '').trim();
      docChildren.push(
        new Paragraph({
          spacing: { before: 60, after: 200 },
          children: [
            new TextRun({ text: 'Keywords—', bold: true, italic: true, font: 'Times New Roman', size: 20 }),
            new TextRun({ text: kw, italic: true, font: 'Times New Roman', size: 20 }),
          ],
        })
      );
      continue;
    } else if (line.trim().length > 0) {
      const cleanAbs = line.replace(/`([^`]+)`/g, '$1').replace(/\$([^\$]+)\$/g, '$1');
      docChildren.push(
        new Paragraph({
          spacing: { after: 100 },
          indent: { left: 360, right: 360 },
          children: [
            new TextRun({
              text: cleanAbs,
              italic: true,
              size: 20, // 10pt
              font: 'Times New Roman',
            }),
          ],
        })
      );
      continue;
    }
  }

  // Headings
  if (line.startsWith('## ')) {
    const title = line.replace('## ', '').trim();
    docChildren.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        spacing: { before: 280, after: 120 },
        children: [
          new TextRun({
            text: title,
            bold: true,
            size: 28, // 14pt
            font: 'Times New Roman',
            color: '002B49',
          }),
        ],
      })
    );
    continue;
  }

  if (line.startsWith('### ')) {
    const title = line.replace('### ', '').trim();
    docChildren.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        spacing: { before: 200, after: 80 },
        children: [
          new TextRun({
            text: title,
            bold: true,
            size: 24, // 12pt
            font: 'Times New Roman',
            color: '1B263B',
          }),
        ],
      })
    );
    continue;
  }

  if (line.startsWith('#### ')) {
    const title = line.replace('#### ', '').trim();
    docChildren.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_3,
        spacing: { before: 140, after: 60 },
        children: [
          new TextRun({
            text: title,
            bold: true,
            italic: true,
            size: 22, // 11pt
            font: 'Times New Roman',
          }),
        ],
      })
    );
    continue;
  }

  // Bullet Points
  if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
    const itemText = line.trim().substring(2).replace(/\*\*/g, '').replace(/`([^`]+)`/g, '$1');
    docChildren.push(
      new Paragraph({
        bullet: { level: 0 },
        spacing: { after: 60 },
        children: [
          new TextRun({
            text: itemText,
            size: 22,
            font: 'Times New Roman',
          }),
        ],
      })
    );
    continue;
  }

  // Regular Paragraphs
  if (line.trim().length > 0) {
    const cleanLine = line.replace(/\*\*/g, '').replace(/`([^`]+)`/g, '$1').replace(/\$([^\$]+)\$/g, '$1');
    docChildren.push(
      new Paragraph({
        spacing: { after: 120 },
        children: [
          new TextRun({
            text: cleanLine,
            size: 22, // 11pt
            font: 'Times New Roman',
          }),
        ],
      })
    );
  }
}

const doc = new Document({
  sections: [
    {
      properties: {},
      children: docChildren,
    },
  ],
});

const outputPath = path.join(__dirname, 'Zero-Latency-Edge-RAG-Paper.docx');

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(outputPath, buffer);
  console.log(`Document successfully generated at: ${outputPath}`);
});
