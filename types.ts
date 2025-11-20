export interface BoundingBox {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
}

export interface DetectionResult {
  box: BoundingBox;
  label: string;
  score: number;
}

export enum AppStatus {
  LOADING_MODEL = 'LOADING_MODEL',
  READY = 'READY',
  ANALYZING = 'ANALYZING',
  ERROR = 'ERROR',
}

export interface PipelineType {
  (image: string | HTMLImageElement, options?: any): Promise<DetectionResult[]>;
}