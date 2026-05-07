export interface Landmark {
  name: string;
  x: number;
  y: number;
  z?: number;
  visibility?: number;
}

export interface SkeletonFrame {
  frame_number?: number;
  timestamp_ms?: number;
  landmarks?: Landmark[];
}

export interface PhraseReviewSegmentMeta {
  segment_id: number;
  start_time_ms: number;
  end_time_ms: number;
}
