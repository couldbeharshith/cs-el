import { z } from 'zod';
import { tool } from 'ai';

const generateChartParameters = z.object({
  company_name: z.string().describe('The name of the company to generate a chart for.'),
});

export const generateChart = tool({
  description: 'Generate a chart for a company by fetching data from the screener API.',
  parameters: generateChartParameters,
  execute: async ({ company_name }) => {
    try {
      const response = await fetch('http://127.0.0.1:8080/accelerated_chart_data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ company_name }),
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch chart data: ${response.statusText}`);
      }

      const chartData = await response.json();
      
      // Transform to QuickChart.io format
      const labels = chartData.data.map((d: any) => d.label);
      const values = chartData.data.map((d: any) => d.value);
      
      const quickChartConfig = {
        type: 'line',
        data: {
          labels: labels,
          datasets: [{
            label: chartData.title || 'Price',
            data: values,
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            tension: 0.1,
            fill: false
          }]
        },
        options: {
          responsive: true,
          plugins: {
            title: {
              display: true,
              text: chartData.title || 'Chart'
            },
            legend: {
              display: true
            }
          },
          scales: {
            y: {
              beginAtZero: false
            }
          }
        }
      };
      
      // Generate QuickChart URL
      const chartUrl = `https://quickchart.io/chart?c=${encodeURIComponent(JSON.stringify(quickChartConfig))}`;
      
      return {
        chartType: chartData.chartType,
        data: chartData.data,
        xAxis: chartData.xAxis,
        yAxis: chartData.yAxis,
        title: chartData.title,
        description: chartData.description,
        chartUrl: chartUrl
      };
    } catch (error) {
      console.error('Error generating chart:', error);
      return { error: 'Failed to generate chart data.' };
    }
  },
});
