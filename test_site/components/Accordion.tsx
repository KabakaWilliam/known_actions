'use client';

import { useState } from 'react';
import { logInteraction } from '@/lib/logger';

interface AccordionItem {
  question: string;
  answer: string;
}

interface AccordionProps {
  items: AccordionItem[];
}

export default function Accordion({ items }: AccordionProps) {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  function toggle(index: number) {
    const next = openIndex === index ? null : index;
    setOpenIndex(next);
    logInteraction(`accordion-${next !== null ? 'open' : 'close'}: ${items[index].question}`);
  }

  return (
    <div className="divide-y divide-zinc-200 border border-zinc-200 rounded-xl overflow-hidden">
      {items.map((item, i) => (
        <div key={i} className="bg-white">
          <button
            onClick={() => toggle(i)}
            aria-expanded={openIndex === i}
            className="w-full flex items-center justify-between px-5 py-4 text-left text-sm font-medium text-zinc-800 hover:bg-zinc-50 transition-colors"
          >
            <span>{item.question}</span>
            <span className="ml-4 shrink-0 text-zinc-400 text-lg leading-none">
              {openIndex === i ? '−' : '+'}
            </span>
          </button>
          {openIndex === i && (
            <div className="px-5 pb-4 text-sm text-zinc-600 leading-relaxed">
              {item.answer}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
