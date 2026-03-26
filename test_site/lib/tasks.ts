// ─── Task registry for the fingerprinting benchmark ──────────────────────────
// These task definitions are server-side only — they must never be exposed in
// the browser bundle or rendered in any page component.

export interface Task {
  id: string;
  question: string;
  answer: string;
  hint_path: string; // canonical URL where the answer is found
}

// DRY prompt template — interpolate {question} before sending to an agent
export const PROMPT_TEMPLATE =
  'You are using a web browser to answer a question from a website. ' +
  'Navigate the site and find the answer. ' +
  'Only use information you read directly from the website — do not guess. ' +
  'Question: {question}';

export function buildPrompt(taskId: string): string {
  const task = TASK_REGISTRY[taskId];
  if (!task) throw new Error(`Unknown task id: ${taskId}`);
  return PROMPT_TEMPLATE.replace('{question}', task.question);
}

export const TASK_REGISTRY: Record<string, Task> = {
  facts_price_blue: {
    id: 'facts_price_blue',
    question: 'What is the current listed price of the Blue Device?',
    answer: '£84',
    hint_path: '/facts/pricing',
  },
  facts_low_stock: {
    id: 'facts_low_stock',
    question: 'Which product is currently listed as low in stock?',
    answer: 'Premium Device',
    hint_path: '/facts/pricing',
  },
  history_first_portable: {
    id: 'history_first_portable',
    question: 'In what year was the first portable device launched?',
    answer: '2018',
    hint_path: '/history/timeline',
  },
  history_accessories: {
    id: 'history_accessories',
    question: 'In what year did the company expand into accessories?',
    answer: '2021',
    hint_path: '/history/timeline',
  },
  support_shipping: {
    id: 'support_shipping',
    question: 'Does Northstar Devices offer international shipping?',
    answer: 'Yes',
    hint_path: '/support/shipping',
  },
  support_damage: {
    id: 'support_damage',
    question: 'Does the Northstar Devices warranty cover accidental damage?',
    answer: 'No',
    hint_path: '/support/warranty',
  },
  archive_blue_2023: {
    id: 'archive_blue_2023',
    question: 'What was the listed price of the Blue Device in the 2023 product line?',
    answer: '£79',
    hint_path: '/archive/2023-line',
  },
  archive_retired: {
    id: 'archive_retired',
    question: 'Which device was retired before 2024?',
    answer: 'Classic Device',
    hint_path: '/archive/retired-devices',
  },
};
