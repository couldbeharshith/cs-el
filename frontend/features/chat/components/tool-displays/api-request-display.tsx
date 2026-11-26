"use client";

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

interface ApiRequestDisplayProps {
  data: {
    status?: number;
    statusText?: string;
    headers?: Record<string, string>;
    data?: any;
    error?: string;
    request?: {
        url: string;
        method: string;
        headers?: any;
        body?: any;
    }
  };
}

export const ApiRequestDisplay: React.FC<ApiRequestDisplayProps> = ({ data }) => {
  const { status, statusText, headers, data: responseData, error, request } = data;

  const isSuccess = status && status >= 200 && status < 300;
  const methodColor = {
    GET: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300',
    POST: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300',
    PUT: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-300',
    DELETE: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300',
    PATCH: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300',
  }[request?.method || 'GET'] || 'bg-gray-100 text-gray-800';

  return (
    <div className="space-y-4 w-full max-w-full">
      {/* Request Details */}
      <Card className="border-l-4 border-l-blue-500">
        <CardHeader className="py-3">
          <div className="flex items-center gap-2 flex-wrap">
            <Badge variant="outline" className={`${methodColor} border-0 font-bold`}>
              {request?.method || 'GET'}
            </Badge>
            <code className="text-sm font-mono break-all">{request?.url}</code>
          </div>
        </CardHeader>
        <CardContent className="py-2">
            <Tabs defaultValue="body" className="w-full">
                <TabsList className="h-8">
                    <TabsTrigger value="body" className="text-xs h-7">Body</TabsTrigger>
                    <TabsTrigger value="headers" className="text-xs h-7">Headers</TabsTrigger>
                </TabsList>
                <TabsContent value="body">
                    {request?.body ? (
                        <ScrollArea className="h-32 w-full rounded-md border p-2 bg-muted/50">
                            <pre className="text-xs font-mono">{JSON.stringify(request.body, null, 2)}</pre>
                        </ScrollArea>
                    ) : (
                        <div className="text-xs text-muted-foreground italic p-2">No request body</div>
                    )}
                </TabsContent>
                <TabsContent value="headers">
                    {request?.headers ? (
                        <ScrollArea className="h-32 w-full rounded-md border p-2 bg-muted/50">
                            <pre className="text-xs font-mono">{JSON.stringify(request.headers, null, 2)}</pre>
                        </ScrollArea>
                    ) : (
                        <div className="text-xs text-muted-foreground italic p-2">No request headers</div>
                    )}
                </TabsContent>
            </Tabs>
        </CardContent>
      </Card>

      {/* Response Details */}
      <Card className={`border-l-4 ${error ? 'border-l-red-500' : isSuccess ? 'border-l-green-500' : 'border-l-yellow-500'}`}>
        <CardHeader className="py-3 flex flex-row items-center justify-between">
          <CardTitle className="text-sm font-medium">Response</CardTitle>
          <div className="flex items-center gap-2">
            {status && (
                <Badge variant={isSuccess ? "default" : "destructive"}>
                    {status} {statusText}
                </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent className="py-2">
            {error ? (
                <div className="text-red-500 text-sm p-2 bg-red-50 dark:bg-red-950/20 rounded">
                    {error}
                </div>
            ) : (
                <Tabs defaultValue="data" className="w-full">
                    <TabsList className="h-8">
                        <TabsTrigger value="data" className="text-xs h-7">Data</TabsTrigger>
                        <TabsTrigger value="headers" className="text-xs h-7">Headers</TabsTrigger>
                    </TabsList>
                    <TabsContent value="data">
                        <ScrollArea className="h-64 w-full rounded-md border p-2 bg-muted/50">
                            <pre className="text-xs font-mono whitespace-pre-wrap break-words">
                                {typeof responseData === 'object' ? JSON.stringify(responseData, null, 2) : String(responseData)}
                            </pre>
                        </ScrollArea>
                    </TabsContent>
                    <TabsContent value="headers">
                        <ScrollArea className="h-32 w-full rounded-md border p-2 bg-muted/50">
                            <pre className="text-xs font-mono">{JSON.stringify(headers, null, 2)}</pre>
                        </ScrollArea>
                    </TabsContent>
                </Tabs>
            )}
        </CardContent>
      </Card>
    </div>
  );
};
