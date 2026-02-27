export type TtsAbbreviationRule = {
  match: string;
  replaceWith: string;
};

// Add or edit rules here. The `match` value is matched as a whole token (case-insensitive).
export const TTS_ABBREVIATION_RULES: TtsAbbreviationRule[] = [
  
  { match: "Mr", replaceWith: "Mister" },
  { match: "Mr.", replaceWith: "Mister" },
  { match: "Mrs", replaceWith: "Missus" },
  { match: "Mrs.", replaceWith: "Missus" },
  { match: "Ms", replaceWith: "Miss" },
  { match: "Ms.", replaceWith: "Miss" },
  { match: "Dr", replaceWith: "Doctor" },
  { match: "Dr.", replaceWith: "Doctor" },
  { match: "Prof", replaceWith: "Professor" },
  { match: "Prof.", replaceWith: "Professor" },
  { match: "St", replaceWith: "Saint" },
  { match: "St.", replaceWith: "Saint" },
  { match: "Sr", replaceWith: "Senior" },
  { match: "Sr.", replaceWith: "Senior" },
  { match: "Jr", replaceWith: "Junior" },
  { match: "Jr.", replaceWith: "Junior" },
  { match: "vs", replaceWith: "versus" },
  { match: "vs.", replaceWith: "versus" },
  { match: "etc", replaceWith: "et cetera" },
  { match: "etc.", replaceWith: "et cetera" },
  { match: "e.g.", replaceWith: "for example" },
  { match: "i.e.", replaceWith: "that is" },
  { match: "No.", replaceWith: "number" },
];

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function romanToInt(romanRaw: string): number | null {
  const roman = romanRaw.toUpperCase();
  if (!/^[MDCLXVI]+$/.test(roman)) {
    return null;
  }

  const values: Record<string, number> = {
    I: 1,
    V: 5,
    X: 10,
    L: 50,
    C: 100,
    D: 500,
    M: 1000,
  };

  let total = 0;
  for (let i = 0; i < roman.length; i += 1) {
    const current = values[roman[i]];
    const next = i + 1 < roman.length ? values[roman[i + 1]] : 0;

    if (!current) {
      return null;
    }

    if (next > current) {
      total -= current;
    } else {
      total += current;
    }
  }

  // Validate by round-tripping to canonical Roman to avoid invalid forms like IIV.
  const canonical = intToRoman(total);
  if (canonical !== roman) {
    return null;
  }

  return total;
}

function intToRoman(value: number): string {
  const symbols: Array<[number, string]> = [
    [1000, "M"],
    [900, "CM"],
    [500, "D"],
    [400, "CD"],
    [100, "C"],
    [90, "XC"],
    [50, "L"],
    [40, "XL"],
    [10, "X"],
    [9, "IX"],
    [5, "V"],
    [4, "IV"],
    [1, "I"],
  ];

  let n = value;
  let out = "";
  for (const [amount, symbol] of symbols) {
    while (n >= amount) {
      out += symbol;
      n -= amount;
    }
  }
  return out;
}

function intToEnglishUpTo1000(value: number): string {
  if (value < 0 || value > 1000) {
    return String(value);
  }

  if (value === 1000) {
    return "one thousand";
  }

  const ones = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
  ];
  const teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
  const tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

  const underHundred = (n: number) => {
    if (n < 10) return ones[n];
    if (n < 20) return teens[n - 10];
    const t = Math.floor(n / 10);
    const o = n % 10;
    return o === 0 ? tens[t] : `${tens[t]} ${ones[o]}`;
  };

  if (value < 100) {
    return underHundred(value);
  }

  const h = Math.floor(value / 100);
  const rest = value % 100;
  if (rest === 0) {
    return `${ones[h]} hundred`;
  }

  return `${ones[h]} hundred ${underHundred(rest)}`;
}

function matchCaseTemplate(value: string, template: string): string {
  if (template === template.toUpperCase()) {
    return value.toUpperCase();
  }
  if (template === template.toLowerCase()) {
    return value.toLowerCase();
  }

  return value
    .split(" ")
    .map((word) => (word ? word[0].toUpperCase() + word.slice(1).toLowerCase() : word))
    .join(" ");
}

function expandChapterRomanNumerals(text: string): string {
  return text.replace(/\b(chapter)\s+([MDCLXVI]+)(?=($|[\s.,!?;:)\]}'"“”'‘’]))/gi, (fullMatch, chapterWord: string, roman: string) => {
    const value = romanToInt(roman);
    if (value === null || value < 1 || value > 1000) {
      return fullMatch;
    }

    const chapterNumber = matchCaseTemplate(intToEnglishUpTo1000(value), chapterWord);
    return `${chapterWord} ${chapterNumber}`;
  });
}

/**
 * Expands common abbreviations into spoken forms before TTS generation.
 */
export function expandTtsAbbreviations(text: string): string {
  let output = expandChapterRomanNumerals(text);

  for (const rule of TTS_ABBREVIATION_RULES) {
    const literal = escapeRegExp(rule.match);
    const pattern = new RegExp(
      `(^|[\\s"“”'‘’([{])(${literal})(?=($|[\\s.,!?;:)\\]}'"“”'‘’]))`,
      "gi",
    );

    output = output.replace(pattern, (_fullMatch, prefix: string) => `${prefix}${rule.replaceWith}`);
  }

  return output;
}

/**
 * Removes standalone noisy line artifacts often found in OCR/public-domain texts,
 * e.g. lines like: "0023m".
 */
export function removeTtsLineArtifacts(text: string): string {
  return text
    .split(/\r?\n/)
    .filter((line) => {
      const trimmed = line.trim();
      if (!trimmed) {
        return true;
      }

      const isCompactAlphaNumericNoise = /^[A-Za-z0-9]{3,10}$/.test(trimmed);
      const hasDigit = /\d/.test(trimmed);
      const hasLetter = /[A-Za-z]/.test(trimmed);

      // Keep normal text lines; remove short standalone alphanumeric blobs with mixed letters+digits.
      if (isCompactAlphaNumericNoise && hasDigit && hasLetter) {
        return false;
      }

      return true;
    })
    .join("\n");
}
