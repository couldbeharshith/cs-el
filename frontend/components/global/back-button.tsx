"use client";

import { useRouter } from "next/navigation";
import { ArrowLeftSquare } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function BackButton(text: {text: string}){
    const router = useRouter();
    return(
        <Button variant={'outline'} className="mb-2" onClick={()=> router.back()}>
            <ArrowLeftSquare />
                {text.text}
        </Button>
    )
}