import { Fragment, type ReactNode } from 'react'

// 轻量 Markdown 渲染器（零第三方依赖）。
// 支持：## / ### 标题、- 与 数字. 列表、**粗体**、空行分段。
// 仅用于展示受控的演示文案与已清洗文本，不处理不可信的 HTML。

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = []
  const parts = text.split(/(\*\*[^*]+\*\*)/g)
  parts.forEach((part, index) => {
    if (!part) return
    const bold = part.match(/^\*\*([^*]+)\*\*$/)
    if (bold) {
      nodes.push(
        <strong key={`${keyPrefix}-b-${index}`} className="font-bold text-[color:var(--ui-text)]">
          {bold[1]}
        </strong>,
      )
    } else {
      nodes.push(<Fragment key={`${keyPrefix}-t-${index}`}>{part}</Fragment>)
    }
  })
  return nodes
}

type Block =
  | { type: 'heading'; level: 2 | 3; text: string }
  | { type: 'list'; ordered: boolean; items: string[] }
  | { type: 'paragraph'; text: string }

function parseBlocks(markdown: string): Block[] {
  const lines = markdown.replace(/\r\n/g, '\n').split('\n')
  const blocks: Block[] = []
  let listItems: string[] | null = null
  let listOrdered = false
  let paragraph: string[] = []

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push({ type: 'paragraph', text: paragraph.join(' ') })
      paragraph = []
    }
  }
  const flushList = () => {
    if (listItems && listItems.length) {
      blocks.push({ type: 'list', ordered: listOrdered, items: listItems })
    }
    listItems = null
  }

  for (const rawLine of lines) {
    const line = rawLine.trim()
    if (!line) {
      flushParagraph()
      flushList()
      continue
    }
    const heading = line.match(/^(#{2,3})\s+(.*)$/)
    if (heading) {
      flushParagraph()
      flushList()
      blocks.push({ type: 'heading', level: heading[1].length === 3 ? 3 : 2, text: heading[2] })
      continue
    }
    const ordered = line.match(/^\d+\.\s+(.*)$/)
    const unordered = line.match(/^[-*]\s+(.*)$/)
    if (ordered || unordered) {
      flushParagraph()
      const nextOrdered = Boolean(ordered)
      if (!listItems || listOrdered !== nextOrdered) {
        flushList()
        listItems = []
        listOrdered = nextOrdered
      }
      listItems.push((ordered ? ordered[1] : unordered![1]).trim())
      continue
    }
    flushList()
    paragraph.push(line)
  }
  flushParagraph()
  flushList()
  return blocks
}

export function Markdown({ text, className = '' }: { text: string; className?: string }) {
  const blocks = parseBlocks(text || '')
  if (!blocks.length) return null
  return (
    <div className={`space-y-2 text-xs leading-relaxed text-[color:var(--ui-text-muted)] ${className}`}>
      {blocks.map((block, index) => {
        if (block.type === 'heading') {
          const Tag = block.level === 3 ? 'h5' : 'h4'
          return (
            <Tag
              key={`h-${index}`}
              className="mt-1 text-xs font-black text-[color:var(--ui-text)]"
            >
              {renderInline(block.text, `h-${index}`)}
            </Tag>
          )
        }
        if (block.type === 'list') {
          const ListTag = block.ordered ? 'ol' : 'ul'
          return (
            <ListTag
              key={`l-${index}`}
              className={`ml-4 space-y-1 ${block.ordered ? 'list-decimal' : 'list-disc'}`}
            >
              {block.items.map((item, itemIndex) => (
                <li key={`l-${index}-${itemIndex}`}>{renderInline(item, `l-${index}-${itemIndex}`)}</li>
              ))}
            </ListTag>
          )
        }
        return (
          <p key={`p-${index}`}>{renderInline(block.text, `p-${index}`)}</p>
        )
      })}
    </div>
  )
}
