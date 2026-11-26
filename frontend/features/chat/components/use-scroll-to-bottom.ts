import { useEffect, useRef, useState } from "react";

export function useScrollToBottom() {
  const container = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  const [showJump, setShowJump] = useState(false);

  function scrollToBottom() {
    pinned.current = true;
    setShowJump(false);
    const element = container.current;
    if (element) element.scrollTop = element.scrollHeight;
  }

  function onScroll() {
    const element = container.current;
    if (!element) return;
    pinned.current = element.scrollHeight - element.scrollTop - element.clientHeight < 72;
    setShowJump(!pinned.current);
  }

  useEffect(() => {
    const list = content.current;
    if (!list) return;
    const observer = new ResizeObserver(() => {
      const element = container.current;
      if (pinned.current && element) element.scrollTop = element.scrollHeight;
    });
    observer.observe(list);
    return () => observer.disconnect();
  }, []);

  return { container, content, onScroll, showJump, scrollToBottom };
}
