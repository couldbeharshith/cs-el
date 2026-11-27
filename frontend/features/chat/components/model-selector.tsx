"use client"

import * as React from "react"
import { CaretSortIcon, CheckIcon } from "@radix-ui/react-icons"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { Drawer, DrawerContent, DrawerTitle, DrawerTrigger } from "@/components/ui/drawer"
import { useIsMobile } from "@/hooks/use-mobile"

interface Model {
  value: string
  label: string
}

interface ModelSelectorProps {
  models: Model[]
  selectedModel: string
  setSelectedModel: (model: string) => void
}

export function ModelSelector({ models, selectedModel, setSelectedModel }: ModelSelectorProps) {
  const [open, setOpen] = React.useState(false)
  const isMobile = useIsMobile()

  const selectedModelLabel = models.find(model => model.value === selectedModel)?.label
  const modelOptions = (
    <Command label="Search models">
      <CommandInput placeholder="Search model..." className="h-9" />
      <CommandList label="Chat models">
        <CommandEmpty>No model found.</CommandEmpty>
        <CommandGroup>
          {models.map((model) => (
            <CommandItem
              key={model.value}
              value={model.value}
              onSelect={() => {
                setSelectedModel(model.value)
                setOpen(false)
              }}
            >
              {model.label}
              <CheckIcon
                className={cn(
                  "ml-auto h-4 w-4",
                  selectedModel === model.value ? "opacity-100" : "opacity-0"
                )}
              />
            </CommandItem>
          ))}
        </CommandGroup>
      </CommandList>
    </Command>
  )

  if (!isMobile) {
    return (
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="ghost"
            type="button"
            aria-label="Chat model"
            role="combobox"
            aria-expanded={open}
            className="w-auto justify-between text-muted-foreground hover:text-foreground"
          >
            {selectedModelLabel}
            <CaretSortIcon className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[200px] p-0">
          {modelOptions}
        </PopoverContent>
      </Popover>
    )
  }

  return (
    <Drawer open={open} onOpenChange={setOpen}>
      <DrawerTrigger asChild>
        <Button type="button" aria-label="Chat model" variant="ghost" className="w-auto justify-between text-muted-foreground hover:text-foreground">
          {selectedModelLabel}
          <CaretSortIcon className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </DrawerTrigger>
      <DrawerContent aria-describedby={undefined}>
        <DrawerTitle className="sr-only">Choose a chat model</DrawerTitle>
        <div className="mt-4 border-t">
          {modelOptions}
        </div>
      </DrawerContent>
    </Drawer>
  )
}
